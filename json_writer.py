import json

dog_problem = {
    "d_sep": [
        [
            ["light-on"],
            ["bowel-problem"],
            ["family-out", "hear-bark"],
            True
        ],
        [
            ["light-on"],
            ["hear-bark"],
            ["bowel-problem"],
            False
        ],
        [
            ["family-out"],
            ["bowel-problem"],
            ["dog-out", "hear-bark"],
            True
        ],
        [
            ["hear-bark"],
            ["bowel-problem"],
            ["light-on", "family-out"],
            False
        ]
    ]
}

lecture_problem = {
    "d_sep": [
        [
            ["Winter?"],
            ["Slippery Road?"],
            ["Rain?"],
            True
        ],
        [
            ["Winter?"],
            ["Wet Grass?"],
            ["Rain?", "Slippery Road?"],
            False
        ],
        [
            ["Sprinkler?"],
            ["Slippery Road?"],
            ["Winter?"],
            False
        ],
        [
            ["Sprinkler?"],
            ["Rain?"],
            ["Wet Grass?"],
            True
        ]
    ]
}

lecture_problem_2 = {
    "d_sep": [
        [
            ["Y"],
            ["I"],
            ["J", "O"],
            True
        ],
        [
            ["J"],
            ["O"],
            ["Y", "I"],
            False
        ],
        [
            ["X"],
            ["Y"],
            ["J", "I"],
            False
        ],
        [
            ["O"],
            ["I"],
            ["X"],
            True
        ]
    ]
}

with open('test_data\\dog_problem.json', 'w') as infile:
    json.dump(dog_problem, infile, indent=4)

with open('test_data\\lecture_problem.json', 'w') as infile:
    json.dump(lecture_problem, infile, indent=4)

with open('test_data\\lecture_problem_2.json', 'w') as infile:
    json.dump(lecture_problem_2, infile, indent=4)
