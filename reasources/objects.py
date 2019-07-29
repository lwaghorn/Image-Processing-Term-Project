from enum import Enum

movement_threshold = 60


class Direction(Enum):
    UP = 1
    DOWN = 2


class Object:
    history = []

    def __init__(self, i, xi, yi, max_age):
        self.i = i
        self.x = xi
        self.y = yi
        self.history = []
        self.done = False
        self.max_age = max_age
        self.dir = None
        self.pulse = 0
        self.history.append([self.x, self.y])

    def get_id(self):  # For the ID
        return self.i

    def in_contour(self, contour_x, contour_y, contour_width, contour_height):
        return abs(contour_x - self.x) <= contour_width and abs(contour_y - self.y) <= contour_height

    def update_coordinates(self, xn, yn):
        # print(abs(self.history[-1][1]-yn))
        if (abs(self.history[-1][1]-yn) > movement_threshold) or (abs(self.history[-1][0]-xn) > movement_threshold):
            # print("outside")
            return
        self.pulse = 0
        self.x = xn
        self.y = yn
        self.history.append([self.x, self.y])

    def set_done(self):
        self.done = True

    def is_done(self):
        return self.done

    def going_up(self, up_line):
        if self.is_done():
            return False
        if len(self.history) >= 2:
            if self.history[-1][1] < up_line <= self.history[-2][1]:
                self.dir = Direction.UP
                self.done = True
                return True
            else:
                return False
        else:
            return False

    def going_down(self, down_line):
        if self.is_done():
            return False
        if len(self.history) >= 2:
            if self.history[-1][1] > down_line >= self.history[-2][1]:
                self.dir = Direction.DOWN
                self.done = True
                return True
            else:
                return False
        else:
            return False

    def age(self):
        self.pulse += 1

    def check_health(self):
        if self.done:
            return
        if self.pulse > self.max_age:
            self.set_done()
            print("killed", self.i)


