import cv2
import numpy as np


class Contain:
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius
        self.count = 1

    def valid(self, center, radius) -> bool:
        center = np.array(center)
        dis1 = np.linalg.norm(np.array(self.center - center))
        dis2 = np.abs(self.radius - radius)

        if dis1 > 2 or dis2 > 5:
            return False
        return True

    def merge(self, center, radius):
        center = np.array(center)
        self.center = (self.center * self.count + center) / (self.count + 1)
        self.radius = (self.radius * self.count + radius) / (self.count + 1)
        self.count += 1

    def __str__(self):
        return f"count={self.count}\tcenter={np.round(self.center, 2)}\tradius={np.round(self.radius, 2)}"


class FitBase:
    def __init__(self, contour=None, score=None, length=None, area=None, *args, **kwargs):
        self.contour = contour
        self.fit_score = score
        self.length = length
        self.area = area

    def fit(self):
        self.length = round(cv2.arcLength(self.contour, True), 2)  # 获取轮廓长度
        self.area = round(cv2.contourArea(self.contour), 2)  # 获取轮廓面积
        return self

    def __str__(self):
        return f'周长={self.length}\t面积={self.area}\t{self.fit_score}'

    def print(self):
        print(self.__str__())

    def update(self, base):
        self.area = base.area
        self.length = base.length
        return self

    def draw(self, img):
        cv2.drawContours(img, [self.contour], -1, (255, 255, 255), 3)
        return

    def valid(self) -> bool:
        if self.area < 500 or len(self.contour) < 10:
            return False
        return True

    def to_json(self):
        return {
            "contour": self.contour,
            "length": self.length,
            "area": self.area,
            "score": self.fit_score
        }


class FitContain(FitBase):
    def __init__(self, center=None, radius=None, *args, **kwargs):
        super(FitContain, self).__init__(*args, **kwargs)
        self.center = center
        self.radius = radius

    def __str__(self):
        return f'长度={self.length}\t面积={self.area}' \
               f'\t圆\t圆心{np.round(self.center, 2)}\t半径={round(self.radius, 2)}' \
               f'\t拟合度:{np.round(self.fit_score, 4)}'

    def print(self):
        print(self.__str__())

    def draw(self, img):
        super(FitContain, self).draw(img)
        cv2.circle(img, (int(self.center[0]), int(self.center[1])), int(self.radius), (0, 255, 0), 2)

    def valid(self) -> bool:
        if len(self.contour) < 10 or self.area < 100000:
            # logger.debug(f"{len(self.contour)}<10 or {self.area} < 100000")
            return False
        if self.area > 1000000:
            # logger.debug("self.area > 1000000")
            return False

        # (x,y) 代表椭圆中心点的位置, radius 代表半径
        self.center, self.radius = cv2.minEnclosingCircle(self.contour)
        data = np.subtract(np.reshape(self.contour, [self.contour.shape[0], 2]), np.array([self.center]))

        score1 = 1 - round(np.abs((np.linalg.norm(data, axis=1) - self.radius).mean()) / self.radius, 4)
        if score1 < 0.8:
            return False

        score2 = 1 - abs(self.radius * self.radius * np.pi / self.area - 1)
        if score2 < 0.8:
            return False
        return True
