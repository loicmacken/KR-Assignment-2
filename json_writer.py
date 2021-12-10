import json

d_sep = 'd_sep'

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
                True
            ],
            [
                [H],
                [B],
                [L, F],
                False
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
                False
            ],
            [
                [P],
                [R],
                [G, W],
                True
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
                True
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
                False
            ],
            [
                [O],
                [I],
                [X],
                True
            ]
        ]
    }

    with open('test_data\\lecture_example2.json', 'w') as infile:
        json.dump(lecture_example2, infile, indent=4)

write_dog_problem()
write_lecture_example()
write_lecture_example2()
