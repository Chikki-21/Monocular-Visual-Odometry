import cv2
import numpy as np
import matplotlib.pyplot as plt 

class VisualOdometry:

    # State variables 
    def __init__(self,video_path):
        self.cap = cv2.VideoCapture(video_path)

        # Initializing orb 
        self.orb = cv2.ORB_create(nfeatures=200)

        # storaage variables 
        self.prev_gray = None 
        self.prev_kp_points = None 

        # lk parameters 
        self.lk = dict(winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))

        # Defining camera intrinsic 
        self.focal_length = 0.8 *1920
        self.principal_point = (960,540)
        self.k = k = np.array([[self.focal_length, 0, self.principal_point[0]],
                               [0, self.focal_length, self.principal_point[1]],
                               [0, 0, 1]], dtype=np.float32)

        # Defining global position and orientation 
        self.position = np.array([0.0,0.0,0.0])
        self.R_global = np.eye(3)

        # Trajectory
        self.traj_x = []
        self.traj_z = []

    # Read frames 
    def read_frames(self):
        ret,frame = self.cap.read()

        if not ret:
            return None 

        return frame

    # Convert to gray frames
    def gray_frame(self,frame):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        return gray

    # Setecting features for first frame 
    def feature_detection(self,gray,frame):
        kp,_ = self.orb.detectAndCompute(gray,None)
    
        # Convert keypoints to points 
        kp_points = np.array([p.pt for p in kp], dtype=np.float32)

        # Drawing key points
        kp_img = cv2.drawKeypoints(frame,kp,None,color=(0,255.0),flags=0)

        self.prev_gray = gray 
        self.prev_kp_points = kp_points.reshape(-1,1,2)


    # Recovering points 
    def recover_features(self,gray):
        kp, des = self.orb.detectAndCompute(gray, None)
        self.prev_kp_points = np.array([p.pt for p in kp], dtype=np.float32).reshape(-1, 1, 2)

    # Tracking points 
    def track_features(self,gray):
        new_kp_points,status,_ = cv2.calcOpticalFlowPyrLK(self.prev_gray,gray,self.prev_kp_points,None,**self.lk)
        
        if new_kp_points is None:
            return None

        good_new = new_kp_points[status.flatten() == 1]
        good_old = self.prev_kp_points[status.flatten() == 1]

        return good_new,good_old
    
    # Calculating essential matrix
    def estimate_pose(self,good_old,good_new):
        E,e_mask = cv2.findEssentialMat(good_old,good_new,self.k,method=cv2.RANSAC,prob=0.999,threshold=1.0)
        
        if E is not None:
            retval,R,t,e_mask = cv2.recoverPose(E,good_old,good_new,self.k,e_mask) # det(R)=1

        return retval,R,t

    # Calcuate position
    def update_pose(self,retval,R,t):
        if retval > 0:
            self.R_global = self.R_global @ R 
            t_world = self.R_global @ t
            self.position = self.position+0.1*t_world.flatten()

            self.traj_x.append(self.position[0])
            self.traj_z.append(self.position[2])
    
    def draw_trajectory(self,frame,good_new,good_old):
        # Draw frames/visualize
        mask = np.zeros_like(frame)
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()

            mask = cv2.line(mask,(int(a),int(b)),(int(c),int(d)),(0,255,0),3)
            frame = cv2.line(frame,(int(a),int(b)),(int(c),int(d)),(0,255,0),3)

        img =  cv2.add(mask,frame)
        cv2.imshow("Tracking img",img)
    
    def run(self):

        while True:
            frame = self.read_frames()
            if frame is None:
                break 

            gray = self.gray_frame(frame)

            if self.prev_gray is None:
                self.feature_detection(gray,frame)
                continue

            if len(self.prev_kp_points)<50:
                self.recover_features(gray)
                self.prev_gray = gray
                continue

            tracked = self.track_features(gray)
            if tracked is None:
                self.prev_gray = None
                continue 

            good_new,good_old = tracked 

            retval,R,t = self.estimate_pose(good_old,good_new)
            
            self.update_pose(retval,R,t)

            self.draw_trajectory(frame,good_new,good_old)

            self.prev_gray = gray
            self.prev_kp_points = good_new.reshape(-1,1,2)

            if cv2.waitKey(50) & 0xFF==ord('q'):
                break

        # Plot
        plt.figure()
        plt.plot(self.traj_x,self.traj_z)
        plt.xlabel("X(Lateral)")
        plt.ylabel("Z(Forward)")
        plt.axis("equal")
        plt.grid()
        plt.show()

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    camera = VisualOdometry("/home/chikki/space/Open-CV/visual_odometry/car_moving.mp4")
    camera.run()

if __name__ == "__main__":
    main()
