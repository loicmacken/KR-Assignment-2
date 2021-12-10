import json

d_sep = 'd_sep'
min_deg = 'min_degree'
min_fill = 'min_fill'

def write_dog_problem():
    F = 'family-out'
    L = 'light-on'
    B = 'bowel-problem'
    D = 'dog-out'
    H = 'hear-bark'

    dog_problem = {
        d_sep: [
            [
                [L],
                [B],
                [F, H],
                True
            ],
            [
                [L],
                [H],
                [B],
                False
            ],
            [
                [F],
                [B],
                [D, H],
                False
            ],
            [
                [H],
                [B],
                [L, F],
                False
            ]
        ],
        min_deg: [
            [
                [F, L, D],
                [L, F, D]
            ],
            [
                [F, B, D, H],
                [H, B, D, F]
            ],
            [
                [F, B, L, D, H],
                [L, H, F, B, D]
            ]
        ],
        min_fill: [
            [
                [F, L, D],
                [L, F, D]
            ],
            [
                [F, B, D, H],
                [B, H, D, F]
            ],
            [
                [F, B, L, D, H],
                [B, L, F, D, H]
            ]
        ]
    }

    with open('test_data\\dog_problem.json', 'w') as infile:
        json.dump(dog_problem, infile, indent=4)

def write_lecture_example():
    W = 'Winter?'
    P = 'Sprinkler?'
    R = 'Rain?'
    G = 'Wet Grass?'
    S = 'Slippery Road?'

    lecture_example = {
        d_sep: [
            [
                [W],
                [S],
                [R],
                True
            ],
            [
                [W],
                [G],
                [R, S],
                False
            ],
            [
                [P],
                [S],
                [W],
                True
            ],
            [
                [P],
                [R],
                [G, W],
                False
            ]
        ],
        min_deg: [
            [
                [W, R, S],
                [S, W, R]
            ],
            [
                [P, G, R, S],
                [S, G, P, R]
            ],
            [
                [W, P, G, R, S],
                [S, W, P, G, R]
            ]
        ]
    }

    with open('test_data\\lecture_example.json', 'w') as infile:
        json.dump(lecture_example, infile, indent=4)

def write_lecture_example2():
    J = 'J'
    I = 'I'
    Y = 'Y'
    X = 'X'
    O = 'O'

    lecture_example2 = {
        d_sep: [
            [
                [Y],
                [I],
                [J, O],
                False
            ],
            [
                [J],
                [O],
                [Y, I],
                False
            ],
            [
                [X],
                [Y],
                [J, I],
                True
            ],
            [
                [O],
                [I],
                [X],
                False
            ]
        ],
        min_deg: [
            [
                [I, Y, X, O],
                [I, O, Y, X]
            ],
            [
                [J, I, X, O],
                [I, J, X, O]
            ],
            [
                [J, I, Y, X, O],
                [I, J, Y, X, O]
            ]
        ]
    }

    with open('test_data\\lecture_example2.json', 'w') as infile:
        json.dump(lecture_example2, infile, indent=4)

write_dog_problem()
write_lecture_example()
write_lecture_example2()
