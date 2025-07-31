# data_generator.py

import random


def generate_synthetic_data(num_samples=10):
    # On crée des phrases types avec des bénéficiaires fictifs
    beneficiaries = ['Jean Dupont', 'Marie Durant', 'Paul Martin']
    templates = [
        "Le bénéficiaire du contrat est {}.",
        "Ce contrat d'assurance vie a pour bénéficiaire {}.",
        "En cas de décès, {} recevra la somme."
    ]
    data = []
    for _ in range(num_samples):
        name = random.choice(beneficiaries)
        template = random.choice(templates)
        sentence = template.format(name)
        words = sentence.split()
        tags = []
        found = False
        for word in words:
            if name.split()[0] in word and not found:
                tags.append('B-ENTITY')  
                found = True
            elif found and (name.split()[-1] in word or word == name.split()[-1]):
                tags.append('I-ENTITY')  
                found = False
            else:
                tags.append('O')  
        data.append((words, tags))
    return data


if __name__ == "__main__":
    # Vérification rapide
    data = generate_synthetic_data(3)
    for sent, tags in data:
        print(list(zip(sent, tags)))



   