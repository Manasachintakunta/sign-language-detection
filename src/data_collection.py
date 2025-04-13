import cv2
import os
import time
import numpy as np

def collect_data():
    # Create directories
    if not os.path.exists("data/raw"):
        os.makedirs("data/raw")
    
    # List of signs to collect
    signs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
 'hello', 'goodbye', 'sorry', 'thankyou', 'yes', 'no', 'i love you', 'love you']

    
    cap = cv2.VideoCapture(0)
    
    for sign in signs:
        sign_dir = f"data/raw/{sign}"
        if not os.path.exists(sign_dir):
            os.makedirs(sign_dir)
        
        print(f"Collecting data for sign: {sign}")
        print("Press 's' to start capturing")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Flip the frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Display instruction
            cv2.putText(frame, f"Prepare for sign: {sign}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Press 's' to start capturing", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
        # Capture images
        counter = 0
        total_images = 200  # Collect 200 images per sign
        
        print(f"Collecting images for sign {sign}. Press 'q' to stop early.")
        
        while counter < total_images:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Flip the frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Display counter
            cv2.putText(frame, f'Collecting {counter}/{total_images}', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Sign: {sign}', (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Data Collection', frame)
            
            # Save frame
            cv2.imwrite(f"{sign_dir}/img_{counter}.jpg", frame)
            counter += 1
            
            # Small delay between captures
            time.sleep(0.1)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print(f"Collected {counter} images for sign {sign}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()