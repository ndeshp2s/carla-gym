
# class ACTIONS:
#     forward = 0
#     left = 1
#     right = 2
#     forward_left = 3
#     forward_right = 4
#     brake = 5
#     brake_left = 6
#     brake_right = 7
#     no_action = 8
#     accelerate = 9
#     decelerate = 10
#     cont = 11

# ACTION_CONTROL = {
#     0: [0.3, 0, 0],
#     1: [0, 0, -1],
#     2: [0, 0, 1],
#     3: [1, 0, -1],
#     4: [1, 0, 1],
#     5: [0, 1, 0],
#     6: [0, 1, -1],
#     7: [0, 1, 1],
#     8: None,
#     9: [0.01 , 0, 1],
#     10: [-0.01, 0 , 1],
#     11: [0,0,0]
# }

# ACTIONS_NAMES = {
#     0: 'forward',
#     1: 'left',
#     2: 'right',
#     3: 'forward_left',
#     4: 'forward_right',
#     5: 'brake',
#     6: 'brake_left',
#     7: 'brake_right',
#     8: 'no_action',
#     9: 'accelerate',
#     10: 'decelerate',
#     11: 'cont'
# }

class ACTIONS:
    accelerate = 0
    decelerate = 1
    cont = 2
    brake = 3

ACTION_CONTROL = {
    0: [0.3, 0, 0],
    1: [0, 0, -1],
    2: [0, 0, 1],
    3: [1, 0, -1],
}

ACTIONS_NAMES = {
    0: 'accelerate',
    1: 'decelerate',
    2: 'cont',
    3: 'brake',
}