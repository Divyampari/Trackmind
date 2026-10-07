import os
from PIL import Image, ImageDraw, ImageFont

def generate_evidence_images():
    evidence_dir = "evidence"
    os.makedirs(evidence_dir, exist_ok=True)

    incidents = [
        {"id": "INC-001", "name": "incident_001.jpg", "worker": "Worker #3", "type": "Restricted Zone Entry", "zone": "Machine A", "severity": "CRITICAL", "color": (239, 68, 68)},
        {"id": "INC-002", "name": "incident_002.jpg", "worker": "Worker #1", "type": "Prolonged Presence", "zone": "Robotic Zone B", "severity": "WARNING", "color": (245, 158, 11)},
        {"id": "INC-003", "name": "incident_003.jpg", "worker": "Worker #7", "type": "Restricted Zone Entry", "zone": "High Voltage Zone C", "severity": "CRITICAL", "color": (239, 68, 68)},
        {"id": "INC-004", "name": "incident_004.jpg", "worker": "Worker #2", "type": "PPE Violation - No Helmet", "zone": "Machine A", "severity": "WARNING", "color": (245, 158, 11)},
        {"id": "INC-005", "name": "incident_005.jpg", "worker": "Worker #5", "type": "Prolonged Presence", "zone": "Chemical Bay E", "severity": "CRITICAL", "color": (239, 68, 68)},
        {"id": "INC-006", "name": "incident_006.jpg", "worker": "Worker #3", "type": "Restricted Zone Entry", "zone": "Packaging Machine D", "severity": "WARNING", "color": (245, 158, 11)},
        {"id": "INC-007", "name": "incident_007.jpg", "worker": "Worker #10", "type": "Unauthorized Access", "zone": "High Voltage Zone C", "severity": "CRITICAL", "color": (239, 68, 68)},
        {"id": "INC-008", "name": "incident_008.jpg", "worker": "Worker #2", "type": "Prolonged Presence", "zone": "Robotic Zone B", "severity": "WARNING", "color": (245, 158, 11)}
    ]

    width, height = 800, 480

    for item in incidents:
        # Create industrial dark dark slate background image
        img = Image.new('RGB', (width, height), color=(15, 23, 42))
        draw = ImageDraw.Draw(img)

        # Draw grid pattern representing factory floor
        for x in range(0, width, 40):
            draw.line([(x, 0), (x, height)], fill=(30, 41, 59), width=1)
        for y in range(0, height, 40):
            draw.line([(0, y), (width, y)], fill=(30, 41, 59), width=1)

        # Draw zone boundary polygon (striped yellow/red hazard zone)
        zone_box = [150, 100, 650, 400]
        draw.rectangle(zone_box, outline=(234, 179, 8), width=3)
        draw.text((160, 110), f"RESTRICTED ZONE - {item['zone'].upper()}", fill=(234, 179, 8))

        # Draw worker bounding box
        bbox = [300, 180, 480, 380]
        box_color = item['color']
        draw.rectangle(bbox, outline=box_color, width=4)
        
        # Bounding box label header
        draw.rectangle([300, 150, 480, 180], fill=box_color)
        draw.text((310, 155), f"{item['worker']} | {item['severity']}", fill=(255, 255, 255))

        # Header overlay banner
        draw.rectangle([0, 0, width, 45], fill=(30, 41, 59))
        draw.text((20, 12), f"FACTORYGUARD AI - EVIDENCE FRAME SNAPSHOT [{item['id']}]", fill=(56, 189, 248))
        draw.text((width - 220, 12), f"EVENT: {item['type']}", fill=(248, 113, 113) if item['severity'] == "CRITICAL" else (251, 191, 36))

        # Bottom watermark
        draw.rectangle([0, height - 30, width, height], fill=(15, 23, 42))
        draw.text((20, height - 22), "DEMO DATASET - PHASE 3 SIMULATED COMPUTER VISION EVIDENCE SNAPSHOT", fill=(148, 163, 184))

        file_path = os.path.join(evidence_dir, item['name'])
        img.save(file_path, "JPEG")
        print(f"Generated {file_path}")

if __name__ == "__main__":
    generate_evidence_images()
