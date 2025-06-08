# Real-Time Player Tracking with YOLOv11 and ByteTrack

This project uses a fine-tuned YOLOv11 model to detect and track football players in real-time from a 15-second match clip. Each player is assigned a unique ID, and the system attempts to preserve this ID even if a player leaves and later re-enters the frame.

---

## 🔧 Setup Instructions

1. **Clone the repository or copy the project folder.**

2. **Create a Python environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)
