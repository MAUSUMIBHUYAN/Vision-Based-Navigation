Overview
This project implements a GPS-independent autonomous drone landing system using computer vision. The drone detects a colored landing marker via its onboard camera, enabling precise navigation and landing without reliance on external signals like GPS or internet connectivity. This solution is particularly valuable in scenarios where GPS is unavailable or unreliable, such as military operations, disaster zones, or space applications.

Features
Spiral Search Algorithm: Expands the drone's search radius systematically when the landing marker is not immediately visible.
Marker Detection: Uses OpenCV for real-time image processing, including color filtering and contour detection.
Precision Landing: Adjusts the drone's position to center over the marker and performs a controlled vertical descent.
Hardware Integration: Compatible with the DJI Tello drone, leveraging its programmable interface for autonomous control.

For all demo videos, visit the Google Drive folder: https://drive.google.com/drive/folders/1rKlSrs1aT5kpTJHOXoW-QnA5ep1nt-Zh?usp=drive_link

Tools and Technologies
Hardware: DJI Tello Drone (with a downward-facing camera modification)
Programming Language: Python
Libraries:
OpenCV: For image processing and marker detection.
djitellopy: To interface with the Tello drone.
NumPy: For numerical operations and array handling.

Limitations
Single Color Detection: The system detects only a predefined single-colored marker.
Marker Visibility: The marker must remain fully visible; partial occlusion can cause detection failures.
No Obstacle Avoidance: The drone does not detect or avoid obstacles during flight.
Lighting Sensitivity: Performance may degrade in low-light or overexposed conditions.
Basic Spiral Search: The search algorithm is not optimized for highly irregular or dynamic environments.
No Angle Compensation: Assumes a vertical camera view; tilted angles may affect accuracy.
