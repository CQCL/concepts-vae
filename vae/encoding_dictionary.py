concept_domains = ['colour', 'size', 'shape', 'position']

enc_dict = {
    'colour' : {
        'blue': 0,
        'green': 1,
        'red': 2,
        'yellow': 3,
    },

    'size' : {
        'small': 0,
        'medium': 1,
        'large': 2,
    },

    'shape' : {
        'triangle': 0,
        'square': 1,
        'pentagon': 2,
        'hexagon': 3,
        'octagon': 4,
        'circle': 5,
        'star_4': 6,
        'star_5': 7,
        'star_6': 8,
        'spoke_4': 9,
        'spoke_5': 10,
        'spoke_6': 11,
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
        3 : 'yellow',
    },

    'size' : {
        0 : 'small',
        1 : 'medium',
        2 : 'large',
    },

    'shape' : {
        0 : 'triangle',
        1 : 'square',
        2 : 'pentagon',
        3 : 'hexagon',
        4 : 'octagon',
        5 : 'circle',
        6 : 'star_4',
        7 : 'star_5',
        8 : 'star_6',
        9 : 'spoke_4',
        10 : 'spoke_5',
        11 : 'spoke_6',
    },

    'position' : {
        0 : 'top',
        1 : 'centre',
        2 : 'bottom',
    }
}
