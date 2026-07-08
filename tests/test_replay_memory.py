from src.research.replay_memory import ReplayMemory


def test_replay_memory_capacity():
    memory = ReplayMemory(capacity=2, seed=0)
    memory.extend(["a", "b", "c"], domain="camvid")
    assert len(memory) == 2


def test_replay_memory_sample_size():
    memory = ReplayMemory(capacity=3, seed=0)
    memory.extend(["a", "b"], domain="camvid")
    sample = memory.sample(batch_size=5)
    assert len(sample) == 2


def test_replay_memory_domains():
    memory = ReplayMemory(capacity=4, seed=0)
    memory.add("a", domain="camvid")
    memory.add("b", domain="cityscapes")
    assert memory.domains() == ["camvid", "cityscapes"]
