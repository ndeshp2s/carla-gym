import pygame
from pygame.locals import *


class DisplayQValues(object):
    def __init__(self):
        self.size = (500, 240)
        self.clock = pygame.time.Clock()
        self.display = pygame.display
        self.fps = 20

        self.screen = self.display.set_mode(self.size, HWSURFACE | DOUBLEBUF)
        self.display.set_caption("Q values")

        pygame.font.init()
        self.font = pygame.font.Font('Roboto-Light.ttf', 20)
        self.font_i = self.font = pygame.font.Font('Roboto-Light.ttf', 20)
        self.font_i.set_italic(True)
        self.font_b = self.font = pygame.font.Font('Roboto-Light.ttf', 20)
        self.font_b.set_bold(True)
        #self.font_i = self.font_b = self.font
        # self.font_i.set_italic(True)
        # self.font_b.set_bold(True)


        self.GREEN = (0, 153, 76)
        self.ORANGE = (204, 102, 0)
        self.RED = (204, 0, 0)
        self.YELLOW = (204, 204, 0)


    def show(self, q):
        self.screen.fill((200, 200, 200))
        font = pygame.font.Font('Roboto-Light.ttf', 20)

        print(q)
        normalized_q_values = self.normalize_array(q)
        print(normalized_q_values)


        # 0th action
        surface = pygame.Surface((120, 200))
        surface.set_colorkey((0,0,0))
        surface.set_alpha(150)
        pygame.draw.rect(surface, self.GREEN, (0, 200, 150, -normalized_q_values[0]))
        text = font.render("accelerate", True, (0, 0, 0))
        
        self.screen.blit(surface, (2.5, 0))
        self.screen.blit(text, (13, 210))

        # 1st action
        surface = pygame.Surface((120, 200))
        surface.set_colorkey((0,0,0))
        surface.set_alpha(150)
        pygame.draw.rect(surface, self.ORANGE, (0, 200, 150, -normalized_q_values[1]))
        text = font.render("decelerate", True, (0, 0, 0))

        self.screen.blit(surface, (127.5, 0))
        self.screen.blit(text, (143, 210))

        # 2nd action
        surface = pygame.Surface((120, 200))
        surface.set_colorkey((0,0,0))
        surface.set_alpha(150)
        pygame.draw.rect(surface, self.RED, (0, 200, 150, -normalized_q_values[2]))
        text = font.render("brake", True, (0, 0, 0))

        self.screen.blit(surface, (252.5, 0))
        self.screen.blit(text, (273, 210))

        # 4th action
        surface = pygame.Surface((120, 200))
        surface.set_colorkey((0,0,0))
        surface.set_alpha(150)
        pygame.draw.rect(surface, self.YELLOW, (0, 200, 150, -0))
        text = font.render("steer", True, (0, 0, 0))

        self.screen.blit(surface, (377.5, 0))
        self.screen.blit(text, (373, 210))


        #pygame.draw.circle(self.screen, (0, 0, 255), (250, 250), 75)
        self.display.flip()
        self.clock.tick()

    def close(self):
        pygame.quit()

    def normalize_array(self, array):
        old_min = min(array)
        old_max = max(array)

        new_min = 0
        new_max = 100

        old_range = (old_max - old_min)
        new_range = (new_max - new_min)

        norm_array = []
        for i in array:
            new_value = (((i - old_min) * new_range) / old_range) + new_min
            norm_array.append(new_value)

        return norm_array



# import random

# q_disp = DisplayQValues()

# running = True
# while running:
#     q0 = random.randint(-100, 200)
#     q1 = random.randint(0, 200)
#     q2 = random.randint(0, 200)
#     q3 = random.randint(0, 200)
#     q = []
#     q.append(q0)
#     q.append(q1)
#     q.append(q2)
#     q.append(q3)

#     q_disp.show(q = q)
