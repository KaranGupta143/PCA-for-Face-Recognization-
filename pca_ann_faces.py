import os, glob, cv2, numpy as np, matplotlib, warnings, math
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump
from datetime import datetime
np.random.seed(42)
warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50%",
    category=UserWarning,
    module="sklearn.metrics._classification",
)

# ---------------------------
# Config
# ---------------------------
# Point to the directory that contains the person subfolders with images
DATA_DIR = "data/lfw-deepfunneled/lfw-deepfunneled"
IMG_SIZE = (80, 80)         # resize for consistency
TEST_SIZE = 0.2
VAL_SPLIT = 0.2             # internal CV handles validation; this is for train/test
N_COMPONENTS = 150          # PCA components (tune this)
OUTPUT_DIR = "artifacts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Data loading
# ---------------------------
def load_dataset(data_dir, img_size=(80,80), max_people=None, max_images_per_person=None, min_images_per_class=None):
    X, y = [], []
    people = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,d))])
    if max_people is not None:
        people = people[:max_people]

    # First pass: gather images per person
    person_to_images = {}
    for person in people:
        person_dir = os.path.join(data_dir, person)
        paths = sorted(glob.glob(os.path.join(person_dir, "*")))
        if max_images_per_person is not None:
            paths = paths[:max_images_per_person]
        person_to_images[person] = [p for p in paths if os.path.isfile(p)]

    # Filter by min_images_per_class if requested
    if min_images_per_class is not None:
        person_to_images = {p: imgs for p, imgs in person_to_images.items() if len(imgs) >= min_images_per_class}

    class_names = sorted(person_to_images.keys())
    for label, person in enumerate(class_names):
        for p in person_to_images[person]:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
            X.append(img.flatten().astype(np.float32))
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y, class_names

quick_run = os.getenv("QUICK_RUN", "0") == "1"

print("[*] Loading images...")
if quick_run:
    # Ensure enough images per class to support 3-fold CV after a train/test split
    min_images_required = 4  # so that ~80% train has at least 3 samples/class
    X, y, class_names = load_dataset(
        DATA_DIR, IMG_SIZE, max_people=120, max_images_per_person=30, min_images_per_class=min_images_required
    )
else:
    # Ensure enough images per class to support 5-fold CV after a train/test split
    min_images_required = 7
    X, y, class_names = load_dataset(DATA_DIR, IMG_SIZE, min_images_per_class=min_images_required)
assert len(X) > 0, f"No images found in {DATA_DIR}"
print(f"Loaded {len(X)} images across {len(class_names)} classes.")

# ---------------------------
# Train / Test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=42
)

# ---------------------------
# Pipeline: Standardize -> PCA -> MLP
# ---------------------------
pca = PCA(n_components=N_COMPONENTS, whiten=True, random_state=42)
mlp = MLPClassifier(max_iter=400, random_state=42, early_stopping=True, n_iter_no_change=20)
if quick_run:
    mlp.set_params(max_iter=200, n_iter_no_change=10)

pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("pca", pca),
    ("mlp", mlp),
])

# ---------------------------
# Hyperparameter search
# ---------------------------
# Determine feasible number of CV splits based on training class counts
class_counts = np.bincount(y_train)
min_class_count = int(class_counts.min()) if class_counts.size > 0 else 0
desired_splits = 3 if quick_run else 5
n_splits = max(2, min(desired_splits, min_class_count)) if min_class_count > 0 else 2

# Dynamically cap PCA components based on training size and per-fold samples
max_components_allowed_global = max(1, min(X_train.shape[0] - 1, X_train.shape[1]))
fold_train_samples = max(2, int(math.floor(X_train.shape[0] * (n_splits - 1) / n_splits)))
max_components_allowed = max(1, min(max_components_allowed_global, fold_train_samples - 1))
if quick_run:
    candidate_pool = [80, 60, 40, 20, 10, 5, 2, 1]
else:
    candidate_pool = [80, 120, 150, 200]
candidate_components = [c for c in candidate_pool if c <= max_components_allowed]
if len(candidate_components) == 0:
    candidate_components = [max_components_allowed]

param_grid = {
    "pca__n_components": candidate_components,
    "mlp__hidden_layer_sizes": [(256,)] if quick_run else [(256,), (256,128), (512,256)],
    "mlp__alpha": [1e-4] if quick_run else [1e-5, 1e-4, 1e-3],
    "mlp__activation": ["relu"] if quick_run else ["relu", "tanh"],
    "mlp__learning_rate_init": [1e-3] if quick_run else [1e-3, 5e-4],
}

if min_class_count >= 2:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    search = GridSearchCV(
        pipe, param_grid, cv=cv, n_jobs=-1, verbose=1, scoring="accuracy"
    )
    print(f"[*] Running grid search with {n_splits}-fold CV...")
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    print("Best params:", search.best_params_)
else:
    # Fallback: fit a single configuration without CV
    print("[!] Insufficient samples per class for CV; fitting a single configuration.")
    pipe.set_params(pca__n_components=candidate_components[0])
    best_model = pipe.fit(X_train, y_train)



# ---------------------------
# Evaluate
# ---------------------------
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------
# Save artifacts
# ---------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(OUTPUT_DIR, f"pca_ann_faces_{timestamp}.joblib")
dump({
    "model": best_model,
    "class_names": class_names,
    "img_size": IMG_SIZE,
}, model_path)
print(f"Saved model to: {model_path}")

# ---------------------------
# Visualize eigenfaces
# ---------------------------
pca_fit = best_model.named_steps["pca"]
eigenfaces = pca_fit.components_.reshape((-1, IMG_SIZE[1], IMG_SIZE[0]))  # (n_components, H, W)

def show_gallery(images, h, w, title, nrow=3, ncol=6, filename=None):
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.6*ncol, 1.8*nrow))
    fig.suptitle(title)
    for i, ax in enumerate(axes.flat):
        if i >= len(images):
            break
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    if filename is None:
        filename = title.lower().replace(" ", "_") + ".png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

show_gallery(eigenfaces[:18], IMG_SIZE[1], IMG_SIZE[0], "Top Eigenfaces (PCA components)", filename="eigenfaces.png")

# ---------------------------
# Visualize reconstructions
# ---------------------------
def reconstruct_samples(X_src, n=10):
    idx = np.random.choice(len(X_src), size=min(n, len(X_src)), replace=False)
    Z = pca_fit.transform(StandardScaler(with_mean=True, with_std=True).fit(X_train).transform(X_src[idx]))
    X_rec = pca_fit.inverse_transform(Z)
    return X_src[idx].reshape(-1, IMG_SIZE[1], IMG_SIZE[0]), X_rec.reshape(-1, IMG_SIZE[1], IMG_SIZE[0])

Xo, Xr = reconstruct_samples(X_test, n=12)
show_gallery(Xo, IMG_SIZE[1], IMG_SIZE[0], "Originals", filename="originals.png")
show_gallery(Xr, IMG_SIZE[1], IMG_SIZE[0], "Reconstructions from PCA Space", filename="reconstructions.png")
