"""
Contrôle de Spotify avec les yeux (webcam).

Gestes :
  - Clignement LONG des deux yeux (>= 0.7 s)  -> piste suivante
  - Clin d'œil GAUCHE (>= 0.4 s)              -> piste précédente
  - Clin d'œil DROIT  (>= 0.4 s)              -> lecture / pause
  - Touche 'q' dans la fenêtre                -> quitter

Les clignements naturels (< 0.4 s) sont ignorés.

Installation :
    pip install opencv-python mediapipe numpy pyautogui spotipy

Deux modes (variable BACKEND) :
  "keys"    : simule les touches multimédia du clavier (simple, aucun compte API).
              Fonctionne bien sous Windows ; sous macOS/Linux, ça dépend du système.
  "spotipy" : utilise l'API Web de Spotify (compte Premium requis).
              Définir les variables d'environnement :
                SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET,
                SPOTIPY_REDIRECT_URI (ex. http://127.0.0.1:8888/callback)
"""

import time

import cv2
import mediapipe as mp
import numpy as np

# ----------------------------- CONFIGURATION -----------------------------
BACKEND = "keys"          # "keys" ou "spotipy"
EAR_THRESHOLD = 0.20      # en dessous : œil considéré fermé (à ajuster)
LONG_BLINK = 0.7          # durée (s) du clignement long
WINK_MIN = 0.4            # durée (s) du clin d'œil
COOLDOWN = 1.2            # délai (s) entre deux commandes
CAMERA_INDEX = 0
# -------------------------------------------------------------------------

# Indices MediaPipe : p1 (coin ext.), p2, p3 (haut), p4 (coin int.), p5, p6 (bas)
# "RIGHT"/"LEFT" = œil droit/gauche de la personne (pas de l'image)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(landmarks, idx, w, h):
    p = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in idx]
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = 2.0 * np.linalg.norm(p[0] - p[3])
    return vertical / horizontal


# ----------------------------- BACKENDS SPOTIFY ---------------------------
if BACKEND == "spotipy":
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth

    sp = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            scope="user-modify-playback-state user-read-playback-state"
        )
    )

    def do_action(action):
        if action == "next":
            sp.next_track()
        elif action == "previous":
            sp.previous_track()
        elif action == "playpause":
            state = sp.current_playback()
            if state and state.get("is_playing"):
                sp.pause_playback()
            else:
                sp.start_playback()
else:
    import pyautogui

    KEYS = {"next": "nexttrack", "previous": "prevtrack", "playpause": "playpause"}

    def do_action(action):
        pyautogui.press(KEYS[action])


# --------------------------------- MAIN -----------------------------------
def main():
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam.")

    # Durée cumulée de chaque configuration pendant un "épisode" de fermeture
    durations = {"both": 0.0, "left": 0.0, "right": 0.0}
    in_episode = False
    last_time = time.time()
    last_action_time = 0.0
    message, message_until = "", 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        dt = now - last_time
        last_time = now

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            ear_r = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            ear_l = eye_aspect_ratio(lm, LEFT_EYE, w, h)
            closed_r = ear_r < EAR_THRESHOLD
            closed_l = ear_l < EAR_THRESHOLD

            if closed_l or closed_r:
                in_episode = True
                if closed_l and closed_r:
                    durations["both"] += dt
                elif closed_l:
                    durations["left"] += dt
                else:
                    durations["right"] += dt
            elif in_episode:
                # Les deux yeux sont rouverts : on analyse l'épisode
                b, l_, r_ = durations["both"], durations["left"], durations["right"]
                action = None
                if b >= LONG_BLINK and b > max(l_, r_):
                    action = "next"
                elif l_ >= WINK_MIN and l_ > b:
                    action = "previous"
                elif r_ >= WINK_MIN and r_ > b:
                    action = "playpause"

                if action and now - last_action_time > COOLDOWN:
                    try:
                        do_action(action)
                        message = {"next": "Suivant >>",
                                   "previous": "<< Precedent",
                                   "playpause": "Play / Pause"}[action]
                    except Exception as e:  # erreur API / clavier
                        message = f"Erreur: {e}"
                    message_until = now + 1.5
                    last_action_time = now

                durations = {"both": 0.0, "left": 0.0, "right": 0.0}
                in_episode = False

            cv2.putText(frame, f"EAR G:{ear_l:.2f} D:{ear_r:.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Visage non detecte", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if now < message_until:
            cv2.putText(frame, message, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 3)

        cv2.imshow("Eye Spotify (q pour quitter)", cv2.flip(frame, 1))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
