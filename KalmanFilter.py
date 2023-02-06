import numpy as np 
import matplotlib.pyplot as plt

""" 
    Simple Kalman Filter example for a moving point with certain position x and veloxity v 
    state vector x = [x v]', a is acceleration, z is measurment (like odometry)
"""

class KalmanFilter: 
    def __init__(self,initial_x: float, initial_v: float, acceleration_variance: float) -> None: ## variable type annotation and func return type 
        
        self.x = np.array([initial_x, initial_v]) # state vector 
        self.acceleration_variance = acceleration_variance 
        self.P = np.eye(2) # Initial P matrix 

    def predict(self, dt: float) -> None: # predict states with specified time sample 
        # x = F*x
        # P = F*P*F' + Q  ## Q = G*G'*a
        F = np.array([[1, dt], [0, 1]])
        G = np.array([0.5 * dt**2, dt]).reshape((2,1))
        Q = G.dot(G.T)*self.acceleration_variance
        
        new_x = F.dot(self.x) # matrix multiplication
        new_P = F.dot(self.P).dot(F.T) + Q 
        
        self.P = new_P
        self.x = new_x 
    
    def measure_and_update(self, measurment: float, measurment_variance:float): 
        # read measurment of position and update state vector estimation and P matrix 
        # y = z - H*x
        # S = H*P*H' + R
        # K = P*H'*inv(S) # kalman gain
        # x = x + K*y
        # P = (I - K*H)*P

        z = np.array([measurment])
        R = np.array([measurment_variance])
        H = np.array([1,0]).reshape((1,2))
        
        y = z - H.dot(self.x)
        S = H.dot(self.P).dot(H.T) + R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        
        new_x = self.x + K.dot(y)
        new_P = (np.eye(2) - K.dot(H)).dot(self.P)
        self.P = new_P
        self.x = new_x 

    @property 
    def return_matrixP(self) -> np.array:
        return self.P
    @property 
    def return_state_vector(self) -> np.array:
        return self.x
    
    
def main(): 
        
    x = 0.2 
    v = 2.3 
    a_v = 0.5
    real_x = 0.0 
    real_v = 2.0 
    meausure_variance = 0.5
    
    plt.ion()
    plt.figure()
    
    kf = KalmanFilter(initial_x=x, initial_v=v, acceleration_variance=a_v)
    
    """
    for i in range(10): 
        det_before = np.linalg.det(kf.return_matrixP)
        kf.predict(dt=0.1)
        det_after = np.linalg.det(kf.return_matrixP)
        print(det_before)
        print(det_after)
    """
    mus = []
    covs = []
    real_vs = []
    real_xs = []
    
    for i in range(1000): 
        
        #if i !=0 and i% 200 ==0: 
        #    real_v *= 1.1
            
        if i > 500: 
            real_v *= 0.95
        
        covs.append(kf.return_matrixP)
        mus.append(kf.return_state_vector)
        
        kf.predict(dt=0.1)  #### predict 
        
        real_x = real_x + 0.1*real_v
        if i != 0 and i % 20 == 0:
            kf.measure_and_update(measurment=real_x + np.random.randn()*np.sqrt(meausure_variance), #### measure and update 
                                  measurment_variance=meausure_variance)
        real_xs.append(real_x)
        real_vs.append(real_v)
        
    plt.subplot(2,1,1)
    plt.title('Position')
    plt.plot([state[0] for state in mus])
    plt.plot([state[0]-2*np.sqrt(covs[0,0]) for state,covs in zip(mus,covs)],'r--')
    plt.plot([state[0]+2*np.sqrt(covs[0,0]) for state,covs in zip(mus,covs)],'r--')
    plt.plot(real_xs, 'k--')
    
    plt.subplot(2,1,2)
    plt.title('Velocity')
    plt.plot([state[1] for state in mus])
    plt.plot([state[1]-2*np.sqrt(covs[1,1]) for state,covs in zip(mus,covs)],'r--')
    plt.plot([state[1]+2*np.sqrt(covs[1,1]) for state,covs in zip(mus,covs)],'r--')
    plt.plot(real_vs, 'k--')
    
    
    plt.show(block = True)
    #plt.ginput(1)
        
    
if __name__ == "__main__": 
    main() 
    
    