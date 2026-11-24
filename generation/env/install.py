"""Install the Python packages for one environment into the CURRENT interpreter.

  python env/install.py gen      # in the "dfgen" conda env
  python env/install.py reface   # in the "dfref" conda env (create it with dlib first, see README)
"""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def pip(*args):
    cmd = [sys.executable, "-m", "pip", "install", *args]
    print(">", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


which = sys.argv[1] if len(sys.argv) > 1 else ""
if which == "gen":
    pip("-r", os.path.join(HERE, "requirements-gen.txt"))
    # mediapipe 0.10.9 pins an old protobuf; the code patches the one call that changed, so skip its pin
    pip("mediapipe==0.10.9", "--no-deps")
    import importlib; importlib.invalidate_caches()
    subprocess.check_call([sys.executable, "-c",
        "import torch, diffusers, transformers, mediapipe, cv2; "
        "print('OK  torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), "
        "'| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"])
elif which == "reface":
    pip("-r", os.path.join(HERE, "requirements-reface.txt"))
    subprocess.check_call([sys.executable, "-c",
        "import torch, dlib, pytorch_lightning, taming, clip; "
        "print('OK  torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), "
        "'| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"])
else:
    sys.exit(__doc__)
