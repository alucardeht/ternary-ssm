import torch
from .config import config_1m
from .model import ModelV2
from .inference import init_state, model_step, generate


def test_inference_state_init():
    config = config_1m()
    model = ModelV2(config).cuda()

    state = init_state(model, batch_size=2, device=torch.device('cuda'))

    assert state.position == 0
    assert state.v_first is None
    assert len(state.layer_states) == config.n_layer

    for layer_state in state.layer_states:
        assert isinstance(layer_state, dict)

    print("test_inference_state_init: PASSED")


def test_model_step():
    config = config_1m()
    model = ModelV2(config).eval().cuda()

    state = init_state(model, batch_size=2, device=torch.device('cuda'))

    token_id = torch.randint(0, config.vocab_size, (2, 1), device=torch.device('cuda'))

    logits, new_state = model_step(model, token_id, state)

    assert logits.shape == (2, 1, config.vocab_size)
    assert new_state.position == 1
    assert len(new_state.layer_states) == config.n_layer

    print("test_model_step: PASSED")


def test_generate():
    config = config_1m()
    model = ModelV2(config).eval().cuda()

    prompt = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long, device=torch.device('cuda'))

    generated = generate(
        model,
        prompt,
        max_tokens=10,
        temperature=0.8,
        top_k=None,
        top_p=None,
        batch_size=1
    )

    assert generated.shape[0] == 1
    assert generated.shape[1] == 15
    assert generated[0, :5].tolist() == [1, 2, 3, 4, 5]

    print("test_generate: PASSED")


if __name__ == '__main__':
    test_inference_state_init()
    test_model_step()
    test_generate()
    print("\nAll tests passed!")
