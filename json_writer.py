import json

d_sep = 'd_sep'
min_deg = 'min_degree'
min_fill = 'min_fill'
net_prune = 'net_prune'
mar_dist = 'marginal_distrib'
map_mpe = 'map_and_mpe'

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
        ],
        net_prune: [
            [
                [L, F, D],
                [
                    [B, True]
                ],
                [L, B, D, F],
                [
                    [F, L], 
                    [F, D]
                ]
            ],
            [
                [F],
                [
                    [H, False],
                    [B, True]
                ],
                [B, D, H, F],
                [
                    [D, H],
                    [F, D]
                ]
            ]
        ],
        mar_dist: [
            [
                [F, L],
                [
                    [B, True]
                ],
                [0.090, 0.060, 0.043, 0.807]
            ],
            [
                [F, L, D],
                [],
                [0.087, 0.058, 0.013, 0.247, 0.003, 0.002, 0.029, 0.560]
            ]
        ],
        map_mpe: [
            [
                [
                    [B, False]
                ],
                [F, L],
                [0.807],
                [
                    [L, True],
                    [F, True]
                ]
            ],
            [
                [
                    [H, False],
                    [B, True]
                ],
                [],
                [0.168],
                [
                    [F, True],
                    [D, False],
                    [B, True],
                    [H, False],
                    [L, True]
                ]
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
        ],
        min_fill: [
            [
                [W, R, S],
                [W, S, R]
            ],
            [
                [P, G, R, S],
                [G, P, S, R]
            ],
            [
                [W, P, G, R, S],
                [W, P, G, R, S]
            ]
        ],
        net_prune: [
            [
                [W, P, G],
                [
                    [R, True]
                ],
                [W, P, R, G],
                [
                    [W, P], 
                    [W, R],
                    [P, G]
                ]
            ],
            [
                [P],
                [
                    [W, False],
                    [S, False]
                ],
                [W, P, R, S],
                [
                    [R, S]
                ]
            ]
        ],
        mar_dist: [
            [
                [G],
                [
                    [W, True],
                    [R, False]
                ],
                [0.82, 0.18]
            ],
            [
                [P, R, G],
                [],
                [0.186, 0.000, 0.029, 0.265, 0.079, 0.315, 0.006, 0.120]
            ]
        ],
        map_mpe: [
            [
                [
                    [W, True],
                    [P, False]
                ],
                [G, S],
                [0.215],
                [
                    [G, True],
                    [S, True]
                ]
            ],
            [
                [
                    [P, False], 
                    [R, True], 
                    [G, False]
                ],
                [],
                [0.054],
                [
                    [G, False],
                    [R, True],
                    [P, False],
                    [W, True],
                    [S, True]
                ]
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
        ],
        min_fill: [
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
        ],
        net_prune: [
            [
                [J, Y],
                [
                    [I, False]
                ],
                [I, J, Y],
                [
                    [J, Y]
                ]
            ],
            [
                [O],
                [
                    [J, True],
                    [X, False]
                ],
                [I, J, Y, X, O],
                [
                    [I, X], 
                    [Y, O]
                ]
            ]
        ],
        mar_dist: [
            [
                [J, I],
                [
                    [Y, True],
                    [O, False]
                ],
                [0.495, 0.495, 0.005, 0.005]
            ],
            [
                [J, Y, O],
                [],
                [0.005, 0.000, 0.010, 0.485, 0.247, 0.247, 0.000, 0.005]
            ]
        ],
        map_mpe: [
            [
                [
                    [Y, False],
                    [O, False]
                ],
                [I, X],
                [0.233],
                [
                    [X, False],
                    [I, False]
                ]
            ],
            [
                [
                    [J, False], 
                    [Y, False], 
                    [O, True]
                ],
                [],
                [0.000],
                [
                    [O, True],
                    [X, True],
                    [Y, False],
                    [J, False],
                    [I, False]
                ]
            ]
        ]
    }

    with open('test_data\\lecture_example2.json', 'w') as infile:
        json.dump(lecture_example2, infile, indent=4)

write_dog_problem()
write_lecture_example()
write_lecture_example2()
