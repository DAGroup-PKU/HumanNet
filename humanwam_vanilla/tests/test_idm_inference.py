"""Regression coverage for IDM packed-view inference and training mask parity."""
from types import MethodType
import pytest
import torch
import torch.nn.functional as F
from humanwam.models.wan22.action_dit import ActionDiT
from humanwam.models.wan22.fastwam_idm import FastWAMIDM
from humanwam.models.wan22.mot import MoT
from humanwam.models.wan22.wan_video_dit import WanVideoDiT


def _small_packed_model(head_size=(64, 128)) -> FastWAMIDM:
    video_expert = WanVideoDiT(
        hidden_dim=48,
        in_dim=3,
        ffn_dim=96,
        out_dim=3,
        text_dim=16,
        freq_dim=16,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        attn_head_dim=24,
        num_layers=1,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="first_frame_causal",
        num_view_embeddings=0,
    )
    action_expert = ActionDiT(
        hidden_dim=48,
        action_dim=10,
        ffn_dim=96,
        text_dim=16,
        freq_dim=16,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=24,
        num_layers=1,
    )
    mot = MoT(
        mixtures={"video": video_expert, "action": action_expert},
        mot_checkpoint_mixed_attn=False,
    )
    vae = torch.nn.Module()
    vae.model = torch.nn.Module()
    vae.model.encoder = torch.nn.Sequential()
    vae.model.z_dim = 3
    vae.upsampling_factor = 16
    vae.temporal_downsample_factor = 4
    model = FastWAMIDM(
        video_expert=video_expert,
        action_expert=action_expert,
        mot=mot,
        vae=vae,
        text_dim=16,
        device="cpu",
        torch_dtype=torch.float32,
        action_component_num_arms=1,  # This tiny fixture uses one valid EEF10 block.
        video_vae_layout="robotwin_packed_multiview",
        robotwin_packed_head_size=head_size,
        robotwin_packed_wrist_size=(32, 64),
        robotwin_latent_tshape_head_size=(64, 128),
        robotwin_latent_tshape_wrist_size=(32, 64),
        packed_head_reads_wrist0=True,
        packed_wrist_encode_separately=True,
        vae_enable_channels_last_3d=False,
    )
    model.encode_calls = []

    def fake_encode(self, video, tiled=False, **_):
        del tiled
        self.encode_calls.append(tuple(video.shape))
        if video.shape[2] > 1:
            video = video[:, :, [0, -1]]
        batch_size, channels, num_frames, height, width = video.shape
        frames = video.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )
        latents = F.avg_pool2d(frames, kernel_size=16, stride=16)
        return latents.view(
            batch_size, num_frames, channels, height // 16, width // 16
        ).permute(0, 2, 1, 3, 4).contiguous()

    model._encode_video_latents = MethodType(fake_encode, model)
    return model


@pytest.mark.parametrize('premerged', [False, True])
def test_packed_idm_inference_uses_current_wrists_and_matches_training_mask(monkeypatch, premerged):
    monkeypatch.setenv('FASTWAM_SDPA_BACKEND', 'math')
    monkeypatch.setenv('FASTWAM_MIXED_SDPA_BACKEND', 'math')
    torch.manual_seed(12)
    model = _small_packed_model()
    model.latent_tshape_premerged_input = premerged
    image = torch.randn(3, 96, 128) if premerged else torch.randn(3, 3, 64, 128)
    context = torch.randn(1, 5, 16)
    context_mask = torch.ones(1, 5, dtype=torch.bool)
    prepared, _, _ = model._prepare_inference_input_image(image)
    head, wrist = model._encode_robotwin_packed_multiview(prepared.unsqueeze(2 if premerged else 3))
    pre = model._video_pre_dit(x=head.repeat(1, 1, 2, 1, 1), timestep=torch.zeros(1),
                              context=context, context_mask=context_mask, action=None,
                              fuse_vae_embedding_in_latents=True, packed_wrist_latents=wrist)
    n = pre['tokens'].shape[1]
    training_mask = model._build_layout_teacher_forcing_attention_mask(pre, pre, 4, 'cpu')
    inference_mask = model._build_inference_attention_mask(pre, 4)
    torch.testing.assert_close(inference_mask, training_mask[n:, n:])
    kwargs = dict(prompt=None, action_horizon=4, num_video_frames=5, context=context,
                  context_mask=context_mask, num_inference_steps=2, seed=7)
    first = model.infer_action(input_image=image, **kwargs)['action']
    changed = image.clone()
    if premerged:
        changed[:, 64:] += 2
    else:
        changed[1:] += 2
    second = model.infer_action(input_image=changed, **kwargs)['action']
    assert first.shape == (4, 10) and torch.isfinite(first).all()
    assert torch.isfinite(second).all()
    assert not torch.allclose(first, second, atol=1e-6, rtol=1e-6)


def test_video_prefill_tokens_match_video_expert_forward(monkeypatch):
    monkeypatch.setenv('FASTWAM_SDPA_BACKEND', 'math')
    monkeypatch.setenv('FASTWAM_MIXED_SDPA_BACKEND', 'math')
    torch.manual_seed(13)
    model = _small_packed_model().eval()
    kwargs = dict(x=torch.randn(1, 3, 2, 4, 8), timestep=torch.tensor([500.]),
                  context=torch.randn(1, 5, 16), context_mask=torch.ones(1, 5, dtype=torch.bool),
                  action=None, fuse_vae_embedding_in_latents=True)
    pre = model.video_expert.pre_dit(**kwargs)
    mask = model.video_expert.build_video_to_video_mask(
        pre['tokens'].shape[1], pre['meta']['tokens_per_frame'], 'cpu')
    cache, tokens = model.mot.prefill_video_cache(
        pre['tokens'], pre['freqs'], pre['t_mod'],
        {'context': pre['context'], 'mask': pre['context_mask']}, mask,
        return_tokens=True)
    assert len(cache) == model.mot.num_layers
    torch.testing.assert_close(model.video_expert.post_dit(tokens, pre),
                               model.video_expert(**kwargs), atol=1e-5, rtol=1e-5)
