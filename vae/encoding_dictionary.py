concept_domains = ['colour', 'size', 'shape', 'position']

enc_dict = {
    'colour' : {
        'any': -1,
        'blue': 0,
        'green': 1,
        'red': 2,
        'purered': 3,
        'pureblue': 4,
        'puregreen': 5,
    },

    'size' : {
        'any': -1,
        'small': 0,
        'medium': 1,
        'large': 2,
    },

    'shape' : {
        'any': -1,
        'triangle': 0,
        'square': 1,
        'circle': 2,
    },

    'position' : {
        'any': -1,
        'top': 0,
        'centre': 1,
        'bottom': 2,
    }
}

dec_dict = {
    'colour' : {
        -1: 'any',
        0 : 'blue',
        1 : 'green',
        2 : 'red',
        3 : 'purered',
        4 : 'pureblue',
        5 : 'puregreen',
    },

    'size' : {
        -1: 'any',
        0 : 'small',
        1 : 'medium',
        2 : 'large',
    },

    'shape' : {
        -1: 'any',
        0 : 'triangle',
        1 : 'square',
        2 : 'circle',
    },

    'position' : {
        -1: 'any',
        0 : 'top',
        1 : 'centre',
        2 : 'bottom',
    }
}
