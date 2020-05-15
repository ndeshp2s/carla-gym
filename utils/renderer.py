import pygame
from pygame.locals import *
import numpy as np 
import random

class Renderer(object):
    def __init__(self):
        self.size = (1, 1)
        self.screen = None
        self.clock = pygame.time.Clock()
        self.display = pygame.display
        self.fps = 20
        self.pressed_keys = []
        self.is_open = False


    def create_screen(self, width, height):
        """
        Creates a pygame window
        :param width: the width of the window
        :param height: the height of the window
        :return: None
        """
        self.size = (width, height + 100)
        self.screen = self.display.set_mode(self.size, HWSURFACE | DOUBLEBUF)
        self.display.set_caption("Renderer")
        self.is_open = True


    def render_image(self, image):
        """
        Render the given image to the pygame window
        :param image: a grayscale or color image in an arbitrary size. assumes that the channels are the last axis
        :return: None
        """
        if self.is_open:
            if len(image.shape) == 2:
                image = np.stack([image] * 3)
            if len(image.shape) == 3:
                if image.shape[0] == 3 or image.shape[0] == 1:
                    image = np.transpose(image, (1, 2, 0))
            surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
            surface = pygame.transform.scale(surface, self.size)
            self.screen.blit(surface, (0, 0))

            # pygame.font.init()
            # surface1 = pygame.Surface((170, 200))
            # surface1.set_colorkey((0,0,0))
            # surface1.set_alpha(20)
            # pygame.draw.rect(surface1, (0,255,0), (0, 200, 170,-random.randint(0, 200)))
            # self.screen.blit(surface1, (0, 0))
            # font = pygame.font.SysFont(None, 24)
            # img = font.render('accelerate', True, (0, 0, 0))
            # self.screen.blit(img, (50, 205))
            # img = font.render('1.6', True, (0, 200, 0))
            # self.screen.blit(img, (80, 105))

            # BLACK = (0, 0, 0)
            # GREY = (200, 200, 200)
            # GREEN = (0, 255, 0, 0)
            # YELLOW = (255, 0, 255)
            # RED = (255, 0, 0)

            # surface = pygame.Surface((90, 90), pygame.SRCALPHA)
            # pygame.draw.circle(surface,(30,224,33,100),(250,100),10)
            # self.screen.blit(surface, (300,500))

            # rect = Rect(0, 0, 720, 100)
            # pygame.draw.rect(self.screen, GREY, rect)
            # h = random.randint(0, 100)
            # print(h)
            # y = 100
            # rect = Rect(0, y, 170, -h)
            # pygame.draw.rect(self.screen, GREEN, rect)
            # h = random.randint(0, 100)
            # print(h)
            # rect = Rect(180, y, 170, -h)
            # pygame.draw.rect(self.screen, YELLOW, rect)
            # h = random.randint(0, 100)
            # print(h)
            # rect = Rect(360, y, 170, -h)
            # pygame.draw.rect(self.screen, RED, rect, 100)
            # pygame.font.init()
            # font = pygame.font.SysFont(None, 24)
            # img = font.render('accelerate', True, BLACK)
            # self.screen.blit(img, (50, 105))
            # img = font.render('decelerate', True, BLACK)
            # self.screen.blit(img, (220, 105))
            # img = font.render('brake', True, BLACK)
            # self.screen.blit(img, (420, 105))

            self.display.flip()
            self.clock.tick()
            self.get_events()

    def get_events(self):
        """
        Get all the window events in the last tick and reponse accordingly
        :return: None
        """
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                self.pressed_keys.append(event.key)
                # esc pressed
                if event.key == pygame.K_ESCAPE:
                    self.close()
            elif event.type == pygame.KEYUP:
                if event.key in self.pressed_keys:
                    self.pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                self.close()

    def get_key_names(self, key_ids):
        """
        Get the key name for each key index in the list
        :param key_ids: a list of key id's
        :return: a list of key names corresponding to the key id's
        """
        return [pygame.key.name(key_id) for key_id in key_ids]

    def close(self):
        """
        Close the pygame window
        :return: None
        """
        self.is_open = False
        pygame.quit()


