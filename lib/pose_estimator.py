"""Estimate head pose according to the facial landmarks"""
import os
import numpy as np

import cv2

from PIL import Image, ImageDraw, ImageFilter

base = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.normpath(os.path.join(base, '../assets/model.txt'))

class PoseEstimator:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self, img_size=(480, 640)):
        self.size = img_size

        # 3D model points.
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corne
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ]) / 4.5

        self.model_points_68 = self._get_full_model_points()

        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array(
            [[-14.97821226], [-10.62040383], [-2053.03596872]])
        # self.r_vec = None
        # self.t_vec = None

    def _get_full_model_points(self, filename=model_path):
        """Get all 68 3D model points from file"""
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T
        # model_points *= 4
        model_points[:, -1] *= -1

        return model_points

    def solve_pose(self, image_points):
        """
        Solve pose from image points
        Return (rotation_vector, translation_vector) as pose.
        """
        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coeefs)

        # (success, rotation_vector, translation_vector) = cv2.solvePnP(
        #     self.model_points,
        #     image_points,
        #     self.camera_matrix,
        #     self.dist_coeefs,
        #     rvec=self.r_vec,
        #     tvec=self.t_vec,
        #     useExtrinsicGuess=True)
        return (rotation_vector, translation_vector)

    def solve_pose_by_68_points(self, image_points):
        """
        Solve pose from all the 68 image points
        Return (rotation_vector, translation_vector) as pose.
        """

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_68, image_points, self.camera_matrix, self.dist_coeefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_68,
            image_points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)

    def draw_annotation_box(self, image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2, face_mask_img=None):
        """Draw a 3D box as annotation of pose"""
        point_3d = []
        # rear_size = 75
        # rear_depth = 0
        # point_3d.append((-rear_size, -rear_size, rear_depth))
        # point_3d.append((-rear_size, rear_size, rear_depth))
        # point_3d.append((rear_size, rear_size, rear_depth))
        # point_3d.append((rear_size, -rear_size, rear_depth))
        # point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 110
        front_depth = 130
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        # point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # for pt in point_2d:
        #     cv2.circle(image, tuple(pt), 5, (0,0,255), -1)

        base_height, base_width, _ = image.shape
        # print(base_width, base_height)

        if len(face_mask_img.shape) != 3:
            exit()

        mask_height, mask_width, _ = face_mask_img.shape
        # print(mask_height, mask_width)

        pts1 = np.float32([[0,0],[mask_width,0],[0,mask_height],[mask_width,mask_height]])
        pts2 = np.float32([point_2d[0],point_2d[3],point_2d[1],point_2d[2]])
        # pts2 = np.float32([[0,0],[mask_width,0],[0,mask_height],[mask_width,mask_height]])
        # pts2 = np.float32([point_2d[0],point_2d[1],[0,mask_width],[mask_height,mask_width]])
        # pts2 = np.float32([point_2d[0],point_2d[1],point_2d[2],point_2d[3]])

        # pts2 = np.float32([[]point_2d)
        # pts2 = np.float32(point_2d)

        # https://note.nkmk.me/python-pillow-paste/

        mask_img = Image.new("L", (base_width, base_height), 0)
        draw = ImageDraw.Draw(mask_img)
        draw.polygon((tuple(point_2d[0]),tuple(point_2d[1]),tuple(point_2d[2]), tuple(point_2d[3])), fill=255)
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(3))

        M = cv2.getPerspectiveTransform(pts1,pts2)
        face_mask_img = cv2.warpPerspective(face_mask_img, M, (base_width, base_height))

        # # src1 = cv2.imread('lena.png')
        # # src2 = cv2.imread('opencv-logo.png')
        # dst = cv2.addWeighted(dst, 1, image, 0.8, 0)
        # cv2.imwrite("sample.png", dst)

        # face_mask_img = face_mask_img[:, :, ::-1].copy()
        # image = image[:, :, ::-1].copy()

        face_mask_img_pil = Image.fromarray(face_mask_img)
        image_pil = Image.fromarray(image)

        image_pil.paste(face_mask_img_pil, (0, 0), mask_img)
        # image_pil.save('sample.png', quality=95)

        return np.asarray(image_pil)

        # Draw all the lines
        # cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        # cv2.line(image, tuple(point_2d[1]), tuple(
        #     point_2d[6]), color, line_width, cv2.LINE_AA)
        # cv2.line(image, tuple(point_2d[2]), tuple(
        #     point_2d[7]), color, line_width, cv2.LINE_AA)
        # cv2.line(image, tuple(point_2d[3]), tuple(
        #     point_2d[8]), color, line_width, cv2.LINE_AA)

    def get_pose_marks(self, marks):
        """Get marks ready for pose estimation from 68 marks"""
        pose_marks = []
        pose_marks.append(marks[30])    # Nose tip
        pose_marks.append(marks[8])     # Chin
        pose_marks.append(marks[36])    # Left eye left corner
        pose_marks.append(marks[45])    # Right eye right corner
        pose_marks.append(marks[48])    # Left Mouth corner
        pose_marks.append(marks[54])    # Right mouth corner
        return pose_marks
