# test_data_generator.py

from data_generator import generate_synthetic_data


def test_data_generation():
    data = generate_synthetic_data(5)
    assert len(data) == 5  # Vérifie le nombre d'échantillons
    for sent, tags in data:
        assert len(sent) == len(tags)  # Vérifie la correspondance mots/tags
        assert any(tag.startswith('B-') for tag in tags)  # Au moins une entité


if __name__ == "__main__":
    test_data_generation()
    print("Test passed!")
