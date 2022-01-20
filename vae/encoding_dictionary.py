concept_domains = ['colour', 'size', 'shape', 'position']

enc_dict = {
    'colour' : {
        'blue': 0,
        'green': 1,
        'red': 2,
    },

    'size' : {
        'small': 0,
        'medium': 1,
        'large': 2,
    },

    'shape' : {
        'triangle': 0,
        'square': 1,
        'circle': 2,
    },

    'position' : {
        'top': 0,
        'centre': 1,
        'bottom': 2,
    }
}

dec_dict = {
    'colour' : {
        0 : 'blue',
        1 : 'green',
        2 : 'red',
    },

    'size' : {
        0 : 'small',
        1 : 'medium',
        2 : 'large',
    },

    'shape' : {
        0 : 'triangle',
        1 : 'square',
        2 : 'circle',
    },

    'position' : {
        0 : 'top',
        1 : 'centre',
        2 : 'bottom',
    }
}
