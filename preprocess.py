import pandas as pd
import numpy as np
import sys, re

def get_names(func):
    """
    - Extraction du nom de fonction
    - Extraction du nom des paramètres
    """
    if func.strip() == '':
        return np.NaN

    temp = func.split('function')[1].split('(')

    func = temp[0].strip()
    try:
        params = temp[1].split(')')[0].split(',')
    except:
        return np.NaN
    return func, params


def name_split(name):
    """
    - Extrait les différents termes qui composent le nom de la fonction
    - Suppression des charactères avec peu d'importance
    """
    name_ = ''.join([c for c in name if not c.isdigit()])
    if name_[0] == '_':
        name_ = name_[1:]
    if name_[0].islower():
        name_ = name_[0].upper() + name_[1:]

    result = []
    temp = ''
    for w in re.findall('[A-Z][^A-Z]*', name_):
        if len(w) == 1:
            temp += w
        else:
            if temp != '':
                result.append(temp.lower())
                temp = ''
            result.append(w.lower())
    return result


def clean_param_name(name):
    """
    - Nettoie et simplifie les noms de paramètre
    """
    result = name.lower()
    result = ''.join([c for c in result if not (c.isdigit() or c in ['_',' ', '$', '*', '/', ':', '<', '>'])])
    if len(result) == 0 or (len(result) == 1 and result[0].strip() == ''):
        return None
    return result


def vectorize(tokens, bow_name):
    """
    - Prend une liste de mots (termes de fonction / paramètre)
    - La projette dans son bag of words
    - Retourne le vecteur de 1 et 0 qui en résulte
    """
    try:
        with open('bows/'+bow_name, 'r') as file:
            words = [w.rstrip().split('_')[1] for w in file.readlines()]
    except:
        print(f'\n# Bow {bow_name} introuvable')

    if type(tokens) == list:# pour les termes du nom de fonction
        vector = [1 if w in tokens else 0 for w in words]
    else:# pour un paramètre
        vector = [1 if w == tokens else 0 for w in words]
    return vector


def create_inputs(text):
    """
    - Prend l'entrée utilisateur
    - Nettoie et adapte pour appliquer la chaine de traitement
    - Retourne un vecteur lisible par le modèle pour chaque paramètre
    """
    # Extraction
    names = get_names(text)
    if names == np.NaN:
        print('\n# Extraction des noms impossible')
        return None
    print(f"\nFonction {names[0]}, paramètres: {', '.join(names[1])}")

    # Fonction
    func_words = name_split(names[0])
    func_vect = vectorize(func_words, 'f_bow.txt')

    # Paramètres
    params = {n:clean_param_name(n) for n in names[1]}
    # dictionnaire nom d'origine : nom nettoyé
    if params is None:
        print('\n# Aucun paramètre après nettoyage')
        return None

    # Inputs
    n_params = len(params)
    inputs = {}
    for p,clean_p in params.items():
        params_vect = vectorize(clean_p, 'p_bow.txt')
        input_vect = [n_params] + params_vect + func_vect
        inputs[p] = np.array(input_vect).reshape(1, -1)

    # dictionnaire "nom du paramètre" : vecteur lisible par le modèle
    return inputs



if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2:
        print(f"\n# Nom de fichier nécessaire\n")
    else:
        file_name = sys.argv[1]

    try:
        with open(file_name, 'r') as file:
            text = file.read()
    except:
        print(f"\n# Fichier '{file_name}' introuvable\n")
        sys.exit(0)

    inputs = create_inputs(text)
