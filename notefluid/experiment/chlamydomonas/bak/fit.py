import cv2
import numpy as np

from notefluid.experiment.chlamydomonas.bak.domain import FitBase


class FitCircle(FitBase):
    def __init__(self, center=None, radius=None, *args, **kwargs):
        super(FitCircle, self).__init__(*args, **kwargs)
        self.center = center
        self.radius = radius

    def fit(self):
        # (x,y) 代表椭圆中心点的位置, radius 代表半径
        self.center, self.radius = cv2.minEnclosingCircle(self.contour)
        data = np.subtract(np.reshape(self.contour, [self.contour.shape[0], 2]), np.array([self.center]))
        self.fit_score = [1 - round(np.abs((np.linalg.norm(data, axis=1) - self.radius).mean()) / self.radius, 4),
                          1 - abs(self.radius * self.radius * np.pi / self.area - 1)]
        return self

    def __str__(self):
        return f'长度={self.length}\t面积={self.area}' \
               f'\t圆\t圆心{np.round(self.center, 2)}\t半径={round(self.radius, 2)}' \
               f'\t拟合度:{np.round(self.fit_score, 4)}'

    def print(self):
        print(self.__str__())

    def draw(self, img):
        super(FitCircle, self).draw(img)
        cv2.circle(img, (int(self.center[0]), int(self.center[1])), int(self.radius), (0, 255, 0), 2)

    def valid(self) -> bool:
        if self.area < 100000 or self.area > 1000000:
            return False
        if self.fit_score[0] < 0.8:
            return False

        if self.fit_score[1] < 0.8:
            return False
        return True

    def to_json(self):
        res = super(FitCircle, self).to_json()
        res.update({
            "center": self.center,
            "radius": self.radius
        })
        return res


class FitEllipse(FitBase):
    def __init__(self, center=None, radius=None, angle=None, *args, **kwargs):
        super(FitEllipse, self).__init__(*args, **kwargs)
        self.center = center
        self.radius = radius
        self.angle = angle

    def fit(self):
        # (x,y) 代表椭圆中心点的位置
        # (a,b) 代表长短轴长度，应注意a、b为长短轴的直径，而非半径
        # angle 代表了中心旋转的角度
        self.center, self.radius, self.angle = cv2.fitEllipse(self.contour)
        self.fit_score = [1, 1 - abs(self.radius[0] * self.radius[1] * np.pi / 4 / self.area - 1)]
        return self

    def __str__(self):
        return f'长度={self.length}\t面积={self.area}' \
               f'\t椭圆\t圆心{np.round(self.center, 2)}\t半径={np.round(self.radius, 2)}\t角度={round(self.angle, 2)}' \
               f'\t拟合度:{np.round(self.fit_score, 4)}'

    def print(self):
        print(self.__str__())

    def draw(self, img):
        super(FitEllipse, self).draw(img)
        cv2.ellipse(img, (self.center, self.radius, self.angle), (0, 255, 0), 2)

    def valid(self, ellipse_pre=None) -> bool:
        if self.area > 100000:
            return False
        if self.center[0] > 1000 or self.center[0] < 300:
            return False
        if ellipse_pre is not None:
            if np.linalg.norm(np.array(self.center) - np.array(ellipse_pre.center)) > 200:
                return True
                # return False

        if self.fit_score[0] < 0.9:
            return False
        if self.fit_score[1] < 0.3:
            return False
        return True

    def to_json(self):
        res = super(FitEllipse, self).to_json()
        res.update({
            "center": self.center,
            "radius": self.radius,
            "angle": self.angle
        })
        return res
