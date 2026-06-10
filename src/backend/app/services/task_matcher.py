import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from app.models.statistics import TaskStatistics
from app.schemas.task_matcher import MatchResult
from app.core.logging_config import setup_logging

logger = setup_logging()

# The model we use: 'paraphrase-MiniLM-L6-v2'
# It's small, fast, and perfect for short text (task names).
MODEL_NAME = "paraphrase-MiniLM-L6-v2"

# Thresholds for classification
EXACT_THRESHOLD = 0.90
SIMILAR_THRESHOLD = 0.60


class TaskMatcher:
    """
    Singleton service to match task names against existing history.
    Uses SentenceTransformer to generate semantic embeddings.
    """

    def __init__(self):
        self._model = None
        self._loaded = False

    def _load_model(self):
        """
        Loads the AI model into memory.
        We do this lazily (on first use) so the server starts fast.
        """
        if not self._loaded:
            logger.info(f"Loading Task Matching model: {MODEL_NAME}...")
            try:
                # This downloads the model if not present and loads it into RAM
                self._model = SentenceTransformer(MODEL_NAME)
                self._loaded = True
                logger.info("Task Matching model loaded successfully!")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise e

    @property
    def model(self):
        """
        Access the model. If not loaded, load it now.
        This ensures we never try to use a model that isn't ready.
        """
        if not self._loaded:
            self._load_model()
        return self._model

    def find_match(self, db: Session, task_name: str) -> MatchResult:
        """
        Compares a new task name against existing tasks in the database.

        Returns:
            MatchResult with:
            - associated_id: UUID of the matched task (or None)
            - association_status: "same", "similar", or "none"
            - name_vector: The 384-dim vector of the new name (for storage)
        """
        logger.info(f"Starting match for: '{task_name}'")

        # ---------------------------------------------------------
        # 1. Exact String Match (The Fast Path)
        # ---------------------------------------------------------
        # We look for a name that matches exactly (case-insensitive).
        # We use .ilike() for case-insensitive SQL comparison.
        # We strip whitespace to be safe.
        exact_match = (
            db.query(TaskStatistics)
            .filter(TaskStatistics.task_name.ilike(task_name.strip()))
            .first()
        )

        if exact_match:
            logger.info("Found Exact Match!")
            return MatchResult(
                associated_id=exact_match.id,
                association_status="same",
                name_vector=exact_match.task_name_vector,  # Reuse existing vector
            )

        # ---------------------------------------------------------
        # 2. Semantic Similarity (The AI Path)
        # ---------------------------------------------------------

        # Get all historical task names and vectors from DB
        history = db.query(TaskStatistics.id, TaskStatistics.task_name_vector).all()

        # If DB is empty, it's a "none" match
        if not history:
            logger.info("No history found. Status: none")
            # Encode the new name anyway so we can save it later
            new_vector = self.model.encode(
                task_name, normalize_embeddings=True
            ).tolist()
            return MatchResult(
                associated_id=None,
                association_status="none",
                name_vector=new_vector,
            )

        # Encode the NEW task name into a vector (384 dimensions)
        # normalize_embeddings=True makes the vector length 1.0, simplifying cosine similarity
        new_vector = self.model.encode(task_name, normalize_embeddings=True)

        best_score = -1.0
        best_id = None

        # Compare against every existing task
        for stat_id, stored_vector in history:
            if stored_vector is None:
                continue

            # Cosine Similarity Formula: dot(A, B) / (|A| * |B|)
            # Since vectors are normalized (length=1), this simplifies to just dot(A, B)
            # We use numpy for fast math.
            score = np.dot(new_vector, stored_vector)

            if score > best_score:
                best_score = score
                best_id = stat_id

        logger.info(f"Best similarity score: {best_score:.4f}")

        # Classify based on thresholds
        if best_score >= EXACT_THRESHOLD:
            status = "same"
        elif best_score >= SIMILAR_THRESHOLD:
            status = "similar"
        else:
            status = "none"

        logger.info(f"Final Status: {status}")

        return MatchResult(
            associated_id=best_id if status != "none" else None,
            association_status=status,
            name_vector=new_vector.tolist(),  # Convert numpy array to list for DB storage
        )


# Create the single instance that the rest of the app will use
task_matcher = TaskMatcher()
