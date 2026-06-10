import parsedatetime
from datetime import datetime, timedelta
from typing import Optional, Any, Tuple
from uuid import UUID, uuid4
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.models.draft import TaskDraft
from app.models.statistics import (
    TaskStatistics,
    CategoryStatistics,
    TaskStatisticsLocation,
    CategoryStatisticsLocation,
)
from app.models.category import Category
from app.services.task_matcher import task_matcher
from app.core.logging_config import setup_logging

# ============================================================================
# SCHEMAS - Pydantic schemas for validation gates
# ============================================================================
from app.schemas.enrichment import (
    TaskPayloadComputed,
    TaskPayloadComputedWithRefs,
)
from app.schemas.nlp import (
    NlpAddPayload,
    NlpPayloadField,
)
from app.schemas.task_matcher import MatchResult
from app.schemas.task import TaskPayload

logger = setup_logging()


class EnrichmentService:
    """
    Task enrichment service with two-phase execution.
    Phase 1 (Predict):  Overwrite -> DateParse -> DraftSave
    Phase 2 (Commit):   DraftLoad/DraftMerge/ChangeMerge -> Compute

    Public Methods:
        predict_nlp_add()   - NLP add flow (Phase 1 only)
        commit_from_draft()  - Draft commit flow (Phase 2 only)
        commit_manual()      - Manual creation (Phase 1 + 2)
        merge_nlp_modify()   - NLP modify flow
        update_task()        - Update task with computed fields
    """

    # ================================================================
    # MAIN PUBLIC METHODS (with Validation Gates)
    # ================================================================
    #
    # Integration approach (Option B):
    # - Input: Accept both Pydantic schemas (recommended) and dicts (legacy)
    # - Internal: Convert to dict for existing logic
    # - Output: Return dicts (not schemas - to not break callers)
    #
    # The schemas are imported for documentation and optional type hints:
    # - NlpAddPayload: Input for predict_nlp_add()
    # - MatchResult: Input for predict_nlp_add(), commit_manual()
    # - TaskPayload: Input for commit_from_draft(), commit_manual(), merge_nlp_modify(), update_task()
    # - TaskPayloadComputed: Output for update_task()
    # - TaskPayloadComputedWithRefs: Output for commit_from_draft(), commit_manual()
    # ============================================================================

    def predict_nlp_add(
        self,
        db: Session,
        nlp_payload: NlpAddPayload | dict[str, dict[str, Any]],
        match_result: MatchResult | dict[str, Any],
    ) -> Tuple[TaskPayload, UUID]:
        """
        NLP add flow (Phase 1 only).

        Input (Validation Gate):
            nlp_payload: NlpAddPayload or dict with {value, predicted} structure
            match_result: MatchResult or dict from task_matcher.find_match()

        Returns:
            (clean_task_payload, draft_id)
            - clean_task_payload: TaskPayload with resolved dates, overwritten fields
            - draft_id: UUID of saved draft

        Steps:
            1. Parse date strings to datetime (first, for importance calculation)
            2. Determine overwrite map based on match status + datetime deadline
            3. Overwrite predicted fields with historical data
            4. Save to draft table

        Integration Notes:
            - Accept NlpAddPayload schema or legacy dict
            - Accept MatchResult schema or legacy dict
            - Returns dict for backward compatibility with callers
        """
        logger.info(f"Enrichment: predict_nlp_add started")

        # ============================================================================
        # CONVERSION: Schema to dict (Validation Gate)
        # ============================================================================
        # Convert schemas to dicts for internal processing
        if hasattr(nlp_payload, 'model_dump'):
            # It's a NlpAddPayload schema - convert to dict
            nlp_payload_dict = {}
            for field_name in NlpAddPayload.model_fields:
                field_obj = getattr(nlp_payload, field_name)
                nlp_payload_dict[field_name] = {"value": field_obj.value, "predicted": field_obj.predicted}
            nlp_payload = nlp_payload_dict

        # ============================================================================
        # VALIDATION: Validate and set defaults for NLP payload (basic defaults only)
        # ============================================================================
        nlp_payload = self._validate_nlp_add_new(nlp_payload)

        if hasattr(match_result, 'model_dump'):
            # It's a MatchResult schema - convert to dict
            match_result = match_result.model_dump()

        logger.debug(f"  Match status: {match_result.get('association_status')}")

        # First parse dates (flat structure needed for _get_overwrite_map)
        flat_payload = {}
        for field, entry in nlp_payload.items():
            value, _ = self._extract_field(entry)
            flat_payload[field] = value

        # Pre-process fixed_start before date parsing (combine start+fixed_start strings)
        flat_payload = self._pre_process_fixed_start(flat_payload)

        # Parse ALL date fields regardless of fixed_time
        parsed_task = self._date_parse(flat_payload, fixed_time=False)

        # Apply fixed-time rules AFTER date parsing (combine, defaults, etc.)
        parsed_task = self._enforce_fixed_time_rules(parsed_task)

        # Rebuild nlp_payload with parsed datetime for importance calculation
        nlp_payload_with_dates = nlp_payload.copy()
        
        # Include all date fields (rules already resolved state)
        date_fields = ["start", "deadline", "fixed_start"]
        
        for field in date_fields:
            if field in parsed_task and parsed_task[field] is not None:
                nlp_payload_with_dates[field] = {
                    "value": parsed_task[field],
                    "predicted": nlp_payload.get(field, {}).get("predicted", False),
                }

        overwrite_map = self._get_overwrite_map(
            db, match_result, nlp_payload_with_dates
        )

        enriched_task = self._overwrite_fields(parsed_task, overwrite_map)

        draft_id = self._draft_save(db, enriched_task, match_result)

        logger.info(f"Enrichment: predict_nlp_add complete. Draft ID: {draft_id}")

        # Convert output to schema (TaskPayload - no computed, no refs)
        output_schema = self._convert_output(enriched_task, with_computed=False, with_refs=False)
        return output_schema, draft_id

    def commit_from_draft(
        self,
        db: Session,
        request_task: TaskPayload | dict[str, Any],
        draft_id: UUID,
    ) -> TaskPayloadComputedWithRefs | None:
        """
        Draft commit flow (Phase 2 only).

        Input (Validation Gate):
            request_task: TaskPayloadComputedWithRefs or dict from frontend
            draft_id: UUID of saved draft

        Returns:
            full_task_data: Complete task data with internal refs, ready for DB
            - Base fields + computed (urgency, value) + internal refs

        Steps:
            1. Load draft (including match_result) from DB
            2. Merge request with draft (request priority)
            3. Compute urgency/value
            4. Add internal refs (loaded from draft)

        Integration Notes:
            - Accept TaskPayloadComputedWithRefs schema or legacy dict
            - Returns dict for backward compatibility
        """
        logger.info(f"Enrichment: commit_from_draft started for draft '{draft_id}'")

        # ============================================================================
        # CONVERSION: Schema to dict (Validation Gate)
        # ============================================================================
        if hasattr(request_task, 'model_dump'):
            request_task = request_task.model_dump()

        draft_data = self._draft_load(db, draft_id)
        if not draft_data:
            logger.warning(f"Draft {draft_id} not found, returning none")
            return None

        match_result = draft_data.get("match_result", {})

        merged_task = self._draft_merge(request_task, draft_data)

        full_task_data = self._compute(merged_task)

        full_task_data = self._add_internal_refs(full_task_data, match_result)

        logger.info(
            f"Enrichment: commit_from_draft complete. "
            f"Value: {full_task_data.get('value')}, "
            f"Status: {match_result.get('association_status')}"
        )
        logger.debug(f"  full_task_data keys: {list(full_task_data.keys())}")

        # Convert output to schema (TaskPayloadComputedWithRefs - with computed + refs)
        output_schema = self._convert_output(full_task_data, with_computed=True, with_refs=True)
        return output_schema

    def commit_manual(
        self,
        db: Session,
        task_payload: TaskPayload | dict[str, Any],
        match_result: MatchResult | dict[str, Any],
    ) -> TaskPayloadComputedWithRefs:
        """
        Manual creation flow (Phase 1 + 2 combined).

        Input (Validation Gate):
            task_payload: TaskPayload or dict (all explicit fields)
            match_result: MatchResult or dict from task_matcher.find_match()

        Returns:
            full_task_data: Complete task data with internal refs, ready for DB
            - Base fields + computed (urgency, value) + internal refs

        Steps:
            1. Compute urgency/value (no overwrite - all fields explicit)
            2. Add internal refs

        Integration Notes:
            - Accept TaskPayload/MatchResult schemas or legacy dicts
            - Returns dict for backward compatibility
        """
        # ============================================================================
        # CONVERSION: Schema to dict (Validation Gate) - MUST be first!
        # ============================================================================
        if hasattr(task_payload, 'model_dump'):
            task_payload = task_payload.model_dump()

        if hasattr(match_result, 'model_dump'):
            match_result = match_result.model_dump()

        logger.info(
            f"Enrichment: commit_manual started for '{task_payload.get('name')}'"
        )

        full_task_data = self._compute(task_payload)

        full_task_data = self._add_internal_refs(full_task_data, match_result)

        logger.info(
            f"Enrichment: commit_manual complete. "
            f"Value: {full_task_data.get('value')}, "
            f"Status: {match_result.get('association_status')}"
        )

        # Convert output to schema (TaskPayloadComputedWithRefs - with computed + refs)
        output_schema = self._convert_output(full_task_data, with_computed=True, with_refs=True)
        return output_schema

    def merge_nlp_modify(
        self,
        db: Session,
        existing_task: TaskPayload | dict[str, Any],
        changed_fields: dict[str, Any],  # Keep as dict per user request
    ) -> TaskPayload | None:
        """
        NLP modify flow.

        Input (Validation Gate):
            existing_task: TaskPayload or dict (current task from DB)
            changed_fields: dict (fields changed by NLP parse/modify)

        Returns:
            merged_task: Task with merged changes and resolved dates

        Steps:
            1. Parse date strings from changed fields
            2. Apply fixed-time rules on raw NLP output (before merge)
            3. Merge validated changes with existing task
            4. Set defaults for non-temporal fields

        Integration Notes:
            - Accept TaskPayload schema or legacy dict
            - changed_fields stays as dict (dynamic/partial)
            - Returns dict for backward compatibility
        """
        logger.info(f"Enrichment: merge_nlp_modify started")

        # ============================================================================
        # CONVERSION: Schema to dict (Validation Gate)
        # ============================================================================
        if hasattr(existing_task, 'model_dump'):
            existing_task = existing_task.model_dump()

        # Strip noop fixed_time before date parsing (preserve existing temporal state)
        changed_fields = self._strip_noop_fixed_time(changed_fields)

        # Pre-process fixed_start before date parsing (combine start+fixed_start strings)
        changed_fields = self._pre_process_fixed_start(changed_fields)

        # Parse dates from raw NLP output first
        parsed_changed = self._date_parse(changed_fields)

        # Apply fixed-time rules on raw NLP output BEFORE merge
        if "fixed_time" in parsed_changed:
            validated_changed = self._enforce_fixed_time_rules(parsed_changed)
        elif "start" in parsed_changed or "deadline" in parsed_changed:
            validated_changed = parsed_changed.copy()
            validated_changed["fixed_time"] = False
            validated_changed["fixed_start"] = None
            logger.info("Modify: temporal change without fixed_time -> set fixed_time=False, fixed_start=null")
        else:
            validated_changed = parsed_changed

        merged_task = self._change_merge(existing_task, validated_changed)

        validated_task = self._validate_nlp_modify_new(merged_task)

        logger.info(f"Enrichment: merge_nlp_modify complete")

        # Convert output to schema (TaskPayload - no computed, no refs)
        output_schema = self._convert_output(validated_task, with_computed=False, with_refs=False)
        return output_schema

    def update_task(
        self,
        db: Session,
        task_payload: TaskPayload | dict[str, Any],
    ) -> TaskPayloadComputed:
        """
        Update task flow - recalculates computed fields.

        Input (Validation Gate):
            task_payload: TaskPayload or dict (updated task data)

        Returns:
            task_with_computed: Task with recalculated urgency/value

        Integration Notes:
            - Accept TaskPayload schema or legacy dict
        """
        # ============================================================================
        # CONVERSION: Schema to dict (Validation Gate) - MUST be first!
        # ============================================================================
        if hasattr(task_payload, 'model_dump'):
            task_payload = task_payload.model_dump()

        logger.info(f"Enrichment: update_task started for '{task_payload.get('name')}'")

        task_with_computed = self._compute(task_payload)

        logger.info(
            f"Enrichment: update_task complete. Value: {task_with_computed.get('value')}"
        )

        # Convert output to schema (TaskPayloadComputed - with computed, no refs)
        output_schema = self._convert_output(task_with_computed, with_computed=True, with_refs=False)
        return output_schema

    # ================================================================
    # HELPER: EXTRACT FIELD
    # ================================================================

    def _extract_field(self, entry: dict[str, Any]) -> Tuple[Any, bool]:
        """
        Extract value and predicted flag from {value, predicted} structure.

        Input:
            entry: {"value": ..., "predicted": bool} or just raw value

        Returns:
            (value, predicted)
        """
        if isinstance(entry, dict):
            return entry.get("value"), entry.get("predicted", False)
        return entry, False

    # ================================================================
    # HELPER: OVERWRITE DECISION
    # ================================================================

    def _get_overwrite_map(
        self,
        db: Session,
        match_result: dict[str, Any],
        nlp_payload: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """
        Determine which fields to overwrite based on predicted flags and stats.

        Priority chain:
            1. task_statistics (only if records >= 3)
            2. category_statistics (loop through categories by priority)
            3. Keep predicted value

        For duration:
            - If difficulty is predicted=True: use stats difficulty for bucket
            - If difficulty is predicted=False: use actual task_payload difficulty for bucket
        """
        overwrite_map = {}
        stats_id = match_result.get("associated_id")
        categories = self._extract_categories(nlp_payload)

        logger.info(f"Building overwrite map. Stats ID: {stats_id}")

        # First, determine difficulty values (for duration lookup)
        difficulty_predicted = False
        difficulty_from_stats = None
        difficulty_actual = None

        if "difficulty" in nlp_payload:
            diff_value, diff_predicted = self._extract_field(nlp_payload["difficulty"])
            difficulty_predicted = diff_predicted
            difficulty_actual = diff_value

            if diff_predicted:
                # Get difficulty from stats (task or category)
                if stats_id:
                    task_stats = self._get_task_stats(db, stats_id)
                    if task_stats and task_stats.get("records", 0) >= 3:
                        difficulty_from_stats = self._get_value_from_task_stats(
                            task_stats, "difficulty"
                        )

                if difficulty_from_stats is None and categories:
                    difficulty_from_stats = self._get_value_from_category_stats(
                        db, categories, "difficulty"
                    )

        # Get deadline for importance calculation
        deadline = None
        if "deadline" in nlp_payload:
            deadline_value, _ = self._extract_field(nlp_payload["deadline"])
            if isinstance(deadline_value, datetime):
                deadline = deadline_value

        # Get fixed_start for importance calculation (for fixed tasks)
        fixed_start = None
        if "fixed_start" in nlp_payload:
            fixed_start_value, _ = self._extract_field(nlp_payload["fixed_start"])
            if isinstance(fixed_start_value, datetime):
                fixed_start = fixed_start_value

        # Use fixed_start for fixed tasks, deadline otherwise
        deadline_for_importance = fixed_start if fixed_start else deadline

        # Process each field
        fields_to_overwrite = ["difficulty", "duration", "location"]

        for field in fields_to_overwrite:
            if field not in nlp_payload:
                continue

            value, predicted = self._extract_field(nlp_payload[field])
            if not predicted:
                logger.debug(
                    f"  Field '{field}': not predicted, keeping value: {value}"
                )
                continue

            logger.debug(f"  Field '{field}': predicted=True, checking stats")

            overwrite_value = None
            overwrite_source = None

            # === TASK STATISTICS (only if records >= 3) ===
            if stats_id:
                task_stats = self._get_task_stats(db, stats_id)
                if task_stats and task_stats.get("records", 0) >= 3:
                    if field == "difficulty":
                        overwrite_value = self._get_value_from_task_stats(
                            task_stats, field
                        )
                        if overwrite_value is not None:
                            overwrite_source = "task_statistics"
                            logger.info(
                                f"    Overwriting '{field}' from task_statistics: {value} -> {overwrite_value}"
                            )

                    elif field == "duration":
                        # Determine which difficulty to use for bucket
                        if difficulty_predicted:
                            dur_difficulty = difficulty_from_stats
                        else:
                            dur_difficulty = difficulty_actual

                        if dur_difficulty is not None:
                            overwrite_value = self._get_value_from_task_stats(
                                task_stats, field, dur_difficulty
                            )
                            if overwrite_value is not None:
                                overwrite_source = "task_statistics"
                                logger.info(
                                    f"    Overwriting '{field}' from task_statistics: {value} -> {overwrite_value}"
                                )

                    elif field == "importance":
                        base_importance = value
                        overwrite_value = self._calculate_importance(
                            db, base_importance, deadline_for_importance, match_result
                        )
                        if overwrite_value is not None:
                            overwrite_source = "task_statistics"
                            logger.info(
                                f"    Overwriting '{field}' from task_statistics: {value} -> {overwrite_value}"
                            )

                    elif field == "location":
                        location = self._get_location_from_task_stats(db, stats_id)
                        if location:
                            overwrite_value = location
                            overwrite_source = "task_statistics"
                            logger.info(
                                f"    Overwriting '{field}' from task_statistics: {value} -> {overwrite_value}"
                            )

            # === CATEGORY STATISTICS (if no task stats) ===
            if overwrite_value is None and categories:
                if field == "difficulty":
                    overwrite_value = self._get_value_from_category_stats(
                        db, categories, field
                    )
                    if overwrite_value is not None:
                        overwrite_source = "category_statistics"
                        logger.info(
                            f"    Overwriting '{field}' from category_statistics: {value} -> {overwrite_value}"
                        )

                elif field == "duration":
                    # Determine which difficulty to use for bucket
                    if difficulty_predicted:
                        dur_difficulty = difficulty_from_stats
                    else:
                        dur_difficulty = difficulty_actual

                    if dur_difficulty is not None:
                        overwrite_value = self._get_value_from_category_stats(
                            db, categories, field, dur_difficulty
                        )
                        if overwrite_value is not None:
                            overwrite_source = "category_statistics"
                            logger.info(
                                f"    Overwriting '{field}' from category_statistics: {value} -> {overwrite_value}"
                            )

                elif field == "importance":
                    base_importance = value
                    overwrite_value = self._calculate_importance(
                        db, base_importance, deadline_for_importance, match_result
                    )
                    if overwrite_value is not None:
                        overwrite_source = "category_statistics"
                        logger.info(
                            f"    Overwriting '{field}' from category_statistics: {value} -> {overwrite_value}"
                        )

                elif field == "location":
                    location = self._get_location_from_category_stats(db, categories)
                    if location:
                        overwrite_value = location
                        overwrite_source = "category_statistics"
                        logger.info(
                            f"    Overwriting '{field}' from category_statistics: {value} -> {overwrite_value}"
                        )

            if overwrite_value is not None and overwrite_source:
                overwrite_map[field] = {
                    "source": overwrite_source,
                    "data": overwrite_value,
                }
            else:
                logger.warning(
                    f"    No stats found for '{field}', keeping predicted value: {value}"
                )

        return overwrite_map

    def _extract_categories(self, nlp_payload: dict[str, dict[str, Any]]) -> list[str]:
        """Extract category list from payload."""
        if "category" not in nlp_payload:
            return []
        value, _ = self._extract_field(nlp_payload["category"])
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [value]
        return []

    def _get_task_stats(self, db: Session, stats_id: UUID) -> Optional[dict]:
        """Fetch task statistics from DB."""
        if not stats_id:
            return None
        stats = db.query(TaskStatistics).filter(TaskStatistics.id == stats_id).first()
        if stats:
            return {
                "id": stats.id,
                "avg_difficulty": stats.avg_difficulty,
                "avg_difficulty_delta": stats.avg_difficulty_delta,
                "avg_duration": stats.avg_duration,
                "avg_duration_delta": stats.avg_duration_delta,
                "records": stats.records,
                "completed_count": stats.completed_count,
                "uncompleted_count": stats.uncompleted_count,
            }
        return None

    def _calculate_bucket(self, difficulty: float) -> str:
        """Calculate difficulty bucket: round(difficulty * 2) / 2"""
        bucket = round(difficulty * 2) / 2
        return str(bucket)

    def _get_value_from_task_stats(
        self, task_stats: dict, field: str, difficulty: Optional[float] = None
    ) -> Optional[float]:
        """
        Extract difficulty or duration value from task stats.

        For difficulty: avg_difficulty + avg_difficulty_delta
        For duration: uses difficulty bucket to lookup avg_duration[bucket] + delta
        """
        if field == "difficulty":
            avg = task_stats.get("avg_difficulty")
            delta = task_stats.get("avg_difficulty_delta")
            if avg is not None:
                delta = delta if delta is not None else 0.0
                result = avg + delta
                result = max(0.05, min(1.0, result))
                return round(result, 2)
        elif field == "duration":
            if difficulty is None:
                logger.warning("Duration lookup requires difficulty value")
                return None

            bucket = self._calculate_bucket(difficulty)
            duration_map = task_stats.get("avg_duration")
            duration_delta_map = task_stats.get("avg_duration_delta")

            if not duration_map:
                return None

            if bucket in duration_map:
                # Access nested structure: {"count": 5, "avg": 30}
                # Also supports old format for backward compatibility
                avg_bucket = duration_map[bucket]
                if isinstance(avg_bucket, dict):
                    avg_val = avg_bucket.get("avg")
                else:
                    # Old format: direct value
                    avg_val = avg_bucket

                # Duration delta: {"count": 3, "avg": 10}
                delta_bucket = (
                    duration_delta_map.get(bucket) if duration_delta_map else None
                )
                if isinstance(delta_bucket, dict):
                    delta_val = delta_bucket.get("avg", 0)
                else:
                    # Old format: direct value
                    delta_val = delta_bucket if delta_bucket else 0

                if avg_val is not None:
                    result = avg_val + delta_val
                    result = max(5, min(1439, result))
                    return int(result)

            # Bucket not found - return None to let caller try next source
            logger.debug(f"Duration bucket '{bucket}' not found in task_stats")
            return None

        return None

    def _get_value_from_category_stats(
        self,
        db: Session,
        categories: list[str],
        field: str,
        difficulty: Optional[float] = None,
    ) -> Optional[float]:
        """
        Get value from category statistics, looping through categories by priority.

        For difficulty: avg_difficulty + avg_difficulty_delta
        For duration: uses difficulty bucket to lookup avg_duration[bucket] + delta
        """
        for category_name in categories:
            cat_stats = (
                db.query(CategoryStatistics)
                .join(Category, Category.id == CategoryStatistics.category_id)
                .filter(Category.name == category_name)
                .first()
            )

            if not cat_stats:
                logger.warning(
                    f"CategoryStats lookup failed for '{category_name}': no matching category_statistics record"
                )
                continue

            if field == "difficulty":
                avg = cat_stats.avg_difficulty
                delta = cat_stats.avg_difficulty_delta
                if avg is not None:
                    delta = delta if delta is not None else 0.0
                    result = avg + delta
                    result = max(0.05, min(1.0, result))
                    return round(result, 2)

            elif field == "duration":
                if difficulty is None:
                    logger.warning("Duration lookup requires difficulty value")
                    return None

                bucket = self._calculate_bucket(difficulty)
                duration_map = cat_stats.avg_duration or {}
                duration_delta_map = cat_stats.avg_duration_delta or {}

                if bucket in duration_map:
                    # Access nested structure: {"count": 5, "avg": 30}
                    # Also supports old format for backward compatibility
                    avg_bucket = duration_map[bucket]
                    if isinstance(avg_bucket, dict):
                        avg_val = avg_bucket.get("avg")
                    else:
                        # Old format: direct value
                        avg_val = avg_bucket

                    # Duration delta: {"count": 3, "avg": 10}
                    delta_bucket = (
                        duration_delta_map.get(bucket) if duration_delta_map else None
                    )
                    if isinstance(delta_bucket, dict):
                        delta_val = delta_bucket.get("avg", 0)
                    else:
                        # Old format: direct value
                        delta_val = delta_bucket if delta_bucket else 0

                    if avg_val is not None:
                        result = avg_val + delta_val
                        result = max(5, min(1439, result))
                        return int(result)

                # Bucket not found in this category - continue to next category
                logger.debug(
                    f"Duration bucket '{bucket}' not found in category '{category_name}', trying next"
                )
                continue

        return None

    def _get_location_from_task_stats(
        self, db: Session, stats_id: UUID
    ) -> Optional[str]:
        """Get most frequent location from task_statistics_locations."""
        location_record = (
            db.query(TaskStatisticsLocation)
            .filter(TaskStatisticsLocation.statistics_id == stats_id)
            .order_by(desc(TaskStatisticsLocation.count))
            .first()
        )
        if location_record:
            from app.models.location import Location

            location = (
                db.query(Location)
                .filter(Location.id == location_record.location_id)
                .first()
            )
            if location:
                return location.name
        return None

    def _get_location_from_category_stats(
        self, db: Session, categories: list[str]
    ) -> Optional[str]:
        """
        Get most frequent location from category_statistics_locations.
        Loops through categories by priority.
        """
        for category_name in categories:
            cat_stats = (
                db.query(CategoryStatistics)
                .join(Category, Category.id == CategoryStatistics.category_id)
                .filter(Category.name == category_name)
                .first()
            )

            if not cat_stats:
                logger.warning(
                    f"CategoryStats location lookup failed for '{category_name}': no matching record"
                )
                continue

            location_record = (
                db.query(CategoryStatisticsLocation)
                .filter(CategoryStatisticsLocation.statistics_id == cat_stats.id)
                .order_by(desc(CategoryStatisticsLocation.count))
                .first()
            )

            if location_record:
                from app.models.location import Location

                location = (
                    db.query(Location)
                    .filter(Location.id == location_record.location_id)
                    .first()
                )
                if location:
                    return location.name

        return None

    # ================================================================
    # HELPER: OVERWRITE FIELDS
    # ================================================================

    def _overwrite_fields(
        self,
        nlp_payload: dict[str, dict[str, Any]],
        overwrite_map: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Replace predicted fields with historical data.

        Input:
            nlp_payload: Task data with {value, predicted} structure
            overwrite_map: {field: {"source": ..., "data": ...}}

        Returns:
            task: Task with overwritten fields (flat {field: value})
        """
        result = {}

        for field, entry in nlp_payload.items():
            value, predicted = self._extract_field(entry)
            result[field] = value

        for field, config in overwrite_map.items():
            overwrite_value = config["data"]
            source = config["source"]
            del source

            if overwrite_value is not None:
                logger.info(
                    f"Overwriting '{field}': {result.get(field)} -> {overwrite_value}"
                )
                result[field] = overwrite_value

        return result

    # ================================================================
    # HELPER: IMPORTANCE CALCULATION
    # ================================================================

    def _calculate_importance(
        self,
        db: Session,
        base_importance: float,
        deadline: Optional[datetime],
        match_result: dict[str, Any],
    ) -> float:
        """
        Recalculate importance based on deadline proximity and completion rate.

        Formula:
            base = nlp_importance
            deadline_boost = 0.3 if days_left <= 1
                           = 0.2 if days_left <= 3
                           = 0.1 if days_left <= 7
                           = 0 otherwise
            completion_boost = completion_rate * 0.2
            final = min(1.0, base + deadline_boost + completion_boost)
        """
        if deadline is None:
            logger.debug("Importance: no deadline, using base")
            return base_importance

        now = datetime.now()
        days_left = (deadline - now).total_seconds() / 86400

        if days_left <= 1:
            deadline_boost = 0.3
        elif days_left <= 3:
            deadline_boost = 0.2
        elif days_left <= 7:
            deadline_boost = 0.1
        else:
            deadline_boost = 0.0

        completion_rate = self._get_completion_rate(db, match_result)

        completion_boost = completion_rate * 0.2

        final_importance = min(1.0, base_importance + deadline_boost + completion_boost)

        logger.debug(
            f"Importance calculation: base={base_importance}, "
            f"days_left={days_left:.1f}, deadline_boost={deadline_boost}, "
            f"completion_rate={completion_rate}, completion_boost={completion_boost}, "
            f"final={final_importance}"
        )

        return round(final_importance, 2)

    def _get_completion_rate(
        self,
        db: Session,
        match_result: dict[str, Any],
    ) -> float:
        """
        Get completion rate from task or category statistics.

        Returns:
            completion_rate = completed_count / (completed_count + uncompleted_count)
            - From matched task if (completed_count + uncompleted_count) >= 3
            - From category statistics otherwise
            - 0.5 default if no data
        """
        stats_id = match_result.get("associated_id")
        status = match_result.get("association_status", "none")

        if stats_id and status in ("same", "similar"):
            task_stats = self._get_task_stats(db, stats_id)
            if task_stats:
                total = task_stats["completed_count"] + task_stats["uncompleted_count"]
                if total >= 3:
                    completed = task_stats["completed_count"]
                    rate = completed / total if total > 0 else 0.5
                    logger.debug(f"Completion rate from task_stats: {rate}")
                    return rate

        logger.debug("No task stats or insufficient data, checking category stats")
        return 0.5

    # ================================================================
    # HELPER: INTERNAL REFS
    # ================================================================

    def _add_internal_refs(
        self,
        task_data: dict[str, Any],
        match_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Add internal references to task data."""
        result = task_data.copy()
        result["task_statistics_id"] = match_result.get("associated_id")
        result["name_vector"] = match_result.get("name_vector")
        result["association_status"] = match_result.get("association_status")
        return result

    # ================================================================
    # HELPER: VALIDATION FOR NLP ADD PAYLOAD
    # ================================================================

    def _validate_nlp_add_dict(self, nlp_payload: dict) -> dict:
        """
        Validate and set defaults for NLP add payload.

        Rules:
            1. name=None -> "task"
            2. start=None and not fixed_time -> current time
            3. deadline=None and not fixed_time -> 23:59 calculation
            4. difficulty=None -> 0.5
            5. duration=None -> 30
            6. category=None or [] -> []
            7. location=None -> "home"
            8. importance=None -> 0.5
            9. fixed_time=None -> False
            10. fixed_start=None but fixed_time=True -> fixed_time=False

        Input:
            nlp_payload: dict with {value, predicted} structure

        Returns:
            nlp_payload: same structure with validated values
        """
        result = nlp_payload.copy()
        now = datetime.now()

        # Rule 1: name=None -> "task"
        name_entry = result.get("name", {})
        name_value = name_entry.get("value") if name_entry else None
        if name_value is None or name_value == "":
            result["name"] = {"value": "task", "predicted": True}
            logger.info("Validation: name set to 'task' (was None)")
        else:
            # Normalize: capitalize first letter only
            name_value = str(name_value).strip()
            if name_value:
                name_value = name_value[0].upper() + name_value[1:]
            result["name"] = {"value": name_value, "predicted": name_entry.get("predicted", True)}
            logger.info(f"Validation: name normalized to '{name_value}'")

        # Get fixed_time value
        fixed_time_entry = result.get("fixed_time", {})
        fixed_time_value = fixed_time_entry.get("value") if fixed_time_entry else False
        if fixed_time_value is None:
            fixed_time_value = False
            result["fixed_time"] = {"value": False, "predicted": True}
            logger.info("Validation: fixed_time set to False (was None)")

        # Rule 9: fixed_time=None -> False (already handled above)
        # Rule 10: fixed_start=None but fixed_time=True -> fixed_time=False
        if fixed_time_value is True:
            fixed_start_entry = result.get("fixed_start", {})
            fixed_start_value = fixed_start_entry.get("value") if fixed_start_entry else None
            if fixed_start_value is None:
                result["fixed_time"] = {"value": False, "predicted": True}
                logger.debug("Validation: fixed_time set to False (fixed_start was None)")

        # Apply rules for non-fixed_time tasks
        if not fixed_time_value:
            # Rule 2: start=None -> current time
            start_entry = result.get("start", {})
            start_value = start_entry.get("value") if start_entry else None
            if start_value is None:
                current_start = now.replace(second=0, microsecond=0)
                result["start"] = {"value": current_start, "predicted": True}
                logger.debug("Validation: start set to current time (was None)")

            # Rule 3: deadline=None -> calculate
            start_for_deadline = result.get("start", {}).get("value") or now
            if isinstance(start_for_deadline, datetime):
                start_time = start_for_deadline
            else:
                start_time = now

            deadline_entry = result.get("deadline", {})
            deadline_value = deadline_entry.get("value") if deadline_entry else None

            if deadline_value is None:
                # If 23:59 - start >= 7 hours, use today's 23:59
                # Otherwise use tomorrow's 23:59
                today_2359 = start_time.replace(hour=23, minute=59, second=59)
                hours_diff = (today_2359 - start_time).total_seconds() / 3600
                if hours_diff >= 7:
                    deadline = today_2359
                else:
                    deadline = today_2359 + timedelta(days=1)
                result["deadline"] = {"value": deadline, "predicted": True}
                logger.warning("Validation: deadline set based on start time (was None)")

        # Rule 4: difficulty=None -> 0.5
        difficulty_entry = result.get("difficulty", {})
        difficulty_value = difficulty_entry.get("value") if difficulty_entry else None
        if difficulty_value is None:
            result["difficulty"] = {"value": 0.5, "predicted": True}
            logger.warning("Validation: difficulty set to 0.5 (was None)")

        # Rule 5: duration=None -> 30
        duration_entry = result.get("duration", {})
        duration_value = duration_entry.get("value") if duration_entry else None
        if duration_value is None:
            result["duration"] = {"value": 30, "predicted": True}
            logger.warning("Validation: duration set to 30 (was None)")

        # Rule 6: category=None or [] -> []
        category_entry = result.get("category", {})
        category_value = category_entry.get("value") if category_entry else None
        if category_value is None or (isinstance(category_value, list) and len(category_value) == 0):
            result["category"] = {"value": [], "predicted": True}
            logger.warning("Validation: category set to [] (was None or empty)")
        else:
            # Normalize: lowercase each category
            if isinstance(category_value, list):
                category_value = [str(cat).lower().strip() for cat in category_value if cat]
            result["category"] = {"value": category_value, "predicted": category_entry.get("predicted", True)}
            logger.info(f"Validation: category normalized to {category_value}")

        # Rule 7: location=None -> "home"
        location_entry = result.get("location", {})
        location_value = location_entry.get("value") if location_entry else None
        if location_value is None or location_value == "":
            result["location"] = {"value": "home", "predicted": True}
            logger.warning("Validation: location set to 'home' (was None)")
        else:
            # Normalize: lowercase
            location_value = str(location_value).lower().strip()
            result["location"] = {"value": location_value, "predicted": location_entry.get("predicted", True)}
            logger.info(f"Validation: location normalized to '{location_value}'")

        # Rule 8: importance=None -> 0.5
        importance_entry = result.get("importance", {})
        importance_value = importance_entry.get("value") if importance_entry else None
        if importance_value is None:
            result["importance"] = {"value": 0.5, "predicted": True}
            logger.warning("Validation: importance set to 0.5 (was None)")

        return result

    def _validate_nlp_add_new(self, nlp_payload: dict) -> dict:
        """
        Validate and set defaults for NLP add payload (basic defaults only).

        Rules:
            1. name=None -> "task"
            2. difficulty=None -> 0.5
            3. duration=None -> 30
            4. category=None or [] -> []
            5. location=None -> "home"
            6. importance=None -> 0.5
            7. fixed_time=None -> False

        NOTE: fixed_time/fixed_start/start/deadline relationship logic
        is deferred to _enforce_fixed_time_rules() (called after date parse).

        Input:
            nlp_payload: dict with {value, predicted} structure

        Returns:
            nlp_payload: same structure with validated values
        """
        result = nlp_payload.copy()

        # Rule 1: name=None -> "task"
        name_entry = result.get("name", {})
        name_value = name_entry.get("value") if name_entry else None
        if name_value is None or name_value == "":
            result["name"] = {"value": "task", "predicted": True}
            logger.info("Validation: name set to 'task' (was None)")
        else:
            name_value = str(name_value).strip()
            if name_value:
                name_value = name_value[0].upper() + name_value[1:]
            result["name"] = {"value": name_value, "predicted": name_entry.get("predicted", True)}
            logger.info(f"Validation: name normalized to '{name_value}'")

        # Rule 7: fixed_time=None -> False
        fixed_time_entry = result.get("fixed_time", {})
        fixed_time_value = fixed_time_entry.get("value") if fixed_time_entry else False
        if fixed_time_value is None:
            result["fixed_time"] = {"value": False, "predicted": True}
            logger.info("Validation: fixed_time set to False (was None)")

        # Rule 4: difficulty=None -> 0.5
        difficulty_entry = result.get("difficulty", {})
        difficulty_value = difficulty_entry.get("value") if difficulty_entry else None
        if difficulty_value is None:
            result["difficulty"] = {"value": 0.5, "predicted": True}
            logger.warning("Validation: difficulty set to 0.5 (was None)")

        # Rule 5: duration=None -> 30
        duration_entry = result.get("duration", {})
        duration_value = duration_entry.get("value") if duration_entry else None
        if duration_value is None:
            result["duration"] = {"value": 30, "predicted": True}
            logger.warning("Validation: duration set to 30 (was None)")

        # Rule 6: category=None or [] -> []
        category_entry = result.get("category", {})
        category_value = category_entry.get("value") if category_entry else None
        if category_value is None or (isinstance(category_value, list) and len(category_value) == 0):
            result["category"] = {"value": [], "predicted": True}
            logger.warning("Validation: category set to [] (was None or empty)")
        else:
            if isinstance(category_value, list):
                category_value = [str(cat).lower().strip() for cat in category_value if cat]
            result["category"] = {"value": category_value, "predicted": category_entry.get("predicted", True)}
            logger.info(f"Validation: category normalized to {category_value}")

        # Rule 7: location=None -> "home"
        location_entry = result.get("location", {})
        location_value = location_entry.get("value") if location_entry else None
        if location_value is None or location_value == "":
            result["location"] = {"value": "home", "predicted": True}
            logger.warning("Validation: location set to 'home' (was None)")
        else:
            location_value = str(location_value).lower().strip()
            result["location"] = {"value": location_value, "predicted": location_entry.get("predicted", True)}
            logger.info(f"Validation: location normalized to '{location_value}'")

        # Rule 8: importance=None -> 0.5
        importance_entry = result.get("importance", {})
        importance_value = importance_entry.get("value") if importance_entry else None
        if importance_value is None:
            result["importance"] = {"value": 0.5, "predicted": True}
            logger.warning("Validation: importance set to 0.5 (was None)")

        return result

    def _validate_nlp_modify(self, task: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and set defaults for NLP modify payload.

        Rules (same as _validate_nlp_add_dict):
            1. name=None -> "task"
            2. start=None and not fixed_time -> current time
            3. deadline=None and not fixed_time -> 23:59 calculation
            4. difficulty=None -> 0.5
            5. duration=None -> 30
            6. category=None or [] -> []
            7. location=None -> "home"
            8. importance=None -> 0.5
            9. fixed_time=None -> False
            10. fixed_start=None but fixed_time=True -> fixed_time=False

        Input:
            task: dict with plain values (already parsed)

        Returns:
            task: dict with validated values
        """
        result = task.copy()
        now = datetime.now()

        # Rule 1: name=None -> "task"
        name_value = result.get("name")
        if name_value is None or name_value == "":
            result["name"] = "task"
            logger.info("Validation: name set to 'task' (was None)")
        else:
            name_value = str(name_value).strip()
            if name_value:
                name_value = name_value[0].upper() + name_value[1:]
            result["name"] = name_value
            logger.info(f"Validation: name normalized to '{name_value}'")

        # Rule 9: fixed_time=None -> False
        fixed_time_value = result.get("fixed_time")
        if fixed_time_value is None:
            fixed_time_value = False
            result["fixed_time"] = False
            logger.info("Validation: fixed_time set to False (was None)")

        # Rule 10: fixed_start=None but fixed_time=True -> fixed_time=False
        if fixed_time_value is True and result.get("fixed_start") is None:
            result["fixed_time"] = False
            logger.info("Validation: fixed_time set to False (fixed_start was None)")

        # Apply rules for non-fixed_time tasks
        if not fixed_time_value:
            # Rule 2: start=None -> current time
            if result.get("start") is None:
                result["start"] = now.replace(second=0, microsecond=0)
                logger.debug("Validation: start set to current time (was None)")

            # Rule 3: deadline=None -> calculate
            start_time = result.get("start") or now
            if isinstance(start_time, datetime):
                start_for_deadline = start_time
            else:
                start_for_deadline = now

            if result.get("deadline") is None:
                today_2359 = start_for_deadline.replace(hour=23, minute=59, second=59)
                hours_diff = (today_2359 - start_for_deadline).total_seconds() / 3600
                if hours_diff >= 7:
                    deadline = today_2359
                else:
                    deadline = today_2359 + timedelta(days=1)
                result["deadline"] = deadline
                logger.warning("Validation: deadline set based on start time (was None)")

        # Rule 4: difficulty=None -> 0.5
        if result.get("difficulty") is None:
            result["difficulty"] = 0.5
            logger.warning("Validation: difficulty set to 0.5 (was None)")

        # Rule 5: duration=None -> 30
        if result.get("duration") is None:
            result["duration"] = 30
            logger.warning("Validation: duration set to 30 (was None)")

        # Rule 6: category=None or [] -> []
        category_value = result.get("category")
        if category_value is None or (isinstance(category_value, list) and len(category_value) == 0):
            result["category"] = []
            logger.warning("Validation: category set to [] (was None or empty)")
        else:
            if isinstance(category_value, list):
                category_value = [str(cat).lower().strip() for cat in category_value if cat]
            result["category"] = category_value
            logger.info(f"Validation: category normalized to {category_value}")

        # Rule 7: location=None -> "home"
        location_value = result.get("location")
        if location_value is None or location_value == "":
            result["location"] = "home"
            logger.warning("Validation: location set to 'home' (was None)")
        else:
            location_value = str(location_value).lower().strip()
            result["location"] = location_value
            logger.info(f"Validation: location normalized to '{location_value}'")

        # Rule 8: importance=None -> 0.5
        if result.get("importance") is None:
            result["importance"] = 0.5
            logger.warning("Validation: importance set to 0.5 (was None)")

        return result

    def _enforce_fixed_time_consistency(
        self,
        merged_task: dict[str, Any],
        changed_fields: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Enforce mutual exclusivity between fixed_time and start/deadline.

        Rules:
            1. If changed_fields contains fixed_time=True -> set start=None, deadline=None
            2. If changed_fields contains start or deadline -> set fixed_time=False, fixed_start=None

        Input:
            merged_task: dict (task after _change_merge)
            changed_fields: dict (original changed fields for detection)

        Returns:
            dict: task with enforced consistency
        """
        result = merged_task.copy()
        result_changed = {}

        if changed_fields.get("fixed_time") is True:
            result["start"] = None
            result["deadline"] = None
            result_changed["start"] = None
            result_changed["deadline"] = None
            logger.info("Fixed time consistency: set start=None, deadline=None (fixed_time=True)")

        if changed_fields.get("start") is not None or changed_fields.get("deadline") is not None:
            result["fixed_time"] = False
            result["fixed_start"] = None
            result_changed["fixed_time"] = False
            result_changed["fixed_start"] = None
            logger.info("Fixed time consistency: set fixed_time=False, fixed_start=None (start/deadline provided)")

        if result_changed:
            logger.debug(f"Fixed time consistency changes: {result_changed}")

        return result

    def _enforce_fixed_time_rules(self, task: dict) -> dict:
        """
        Enforce fixed-time logic on a flat task dict with parsed datetime objects.

        Rules:
            1. fixed_time is True:
               a. fixed_start not null, deadline not null:
                  -> combine: fixed_start = deadline.date + fixed_start.time
                  -> deadline = null, start = null
               b. fixed_start not null, deadline null:
                  -> start = null
               c. fixed_start null, deadline not null:
                  -> fixed_time = False, fixed_start = null
                  -> if start is null: start = now
               d. fixed_start null, deadline null:
                  -> start = now, deadline = tomorrow 23:59
                  -> fixed_time = False, fixed_start = null
            2. fixed_time is not True:
               -> fixed_start = null
               -> if deadline is null: start = now, deadline = tomorrow 23:59
               -> elif start is null: start = now

        Input:
            task: flat dict with datetime objects (after date parsing)

        Returns:
            task: corrected flat dict
        """
        result = task.copy()
        now = datetime.now().replace(second=0, microsecond=0)

        fixed_time_value = result.get("fixed_time")
        if fixed_time_value is None:
            fixed_time_value = False

        if fixed_time_value is True:
            fixed_start = result.get("fixed_start")
            deadline = result.get("deadline")
            start = result.get("start")

            if fixed_start is not None:
                if deadline is not None:
                    combined = deadline.replace(
                        hour=fixed_start.hour,
                        minute=fixed_start.minute,
                        second=0,
                        microsecond=0,
                    )
                    result["fixed_start"] = combined
                    result["deadline"] = None
                    result["start"] = None
                    logger.info(
                        f"Fixed-time rules: combined deadline ({deadline}) + fixed_start ({fixed_start}) "
                        f"-> fixed_start ({combined}), deadline=null, start=null"
                    )
                else:
                    result["start"] = None
                    logger.info("Fixed-time rules: fixed_start set, deadline null -> start=null")
            else:
                if deadline is not None:
                    result["fixed_time"] = False
                    result["fixed_start"] = None
                    if start is None:
                        result["start"] = now
                        logger.info(
                            "Fixed-time rules: fixed_time=True, fixed_start=null, deadline not null "
                            "-> converted to non-fixed, start=now"
                        )
                    else:
                        logger.info(
                            "Fixed-time rules: fixed_time=True, fixed_start=null, deadline not null "
                            "-> converted to non-fixed, start kept"
                        )
                else:
                    result["start"] = now
                    deadline_dt = now.replace(hour=23, minute=59, second=0, microsecond=0)
                    start_time = start if isinstance(start, datetime) else now
                    hours_diff = (deadline_dt - start_time).total_seconds() / 3600
                    if hours_diff < 7:
                        deadline_dt = deadline_dt + timedelta(days=1)
                    result["deadline"] = deadline_dt
                    result["fixed_time"] = False
                    result["fixed_start"] = None
                    logger.info(
                        "Fixed-time rules: fixed_time=True, fixed_start=null, deadline=null "
                        "-> converted to non-fixed with defaults"
                    )
        else:
            result["fixed_start"] = None
            if result.get("deadline") is None:
                result["start"] = now
                deadline_dt = now.replace(hour=23, minute=59, second=0, microsecond=0)
                start_time = result.get("start", now)
                if not isinstance(start_time, datetime):
                    start_time = now
                hours_diff = (deadline_dt - start_time).total_seconds() / 3600
                if hours_diff < 7:
                    deadline_dt = deadline_dt + timedelta(days=1)
                result["deadline"] = deadline_dt
                logger.info("Fixed-time rules: deadline=null, set default start and deadline")
            elif result.get("start") is None:
                result["start"] = now
                logger.info("Fixed-time rules: start=null, set start=now")
            else:
                logger.info("Fixed-time rules: non-fixed with valid start/deadline, no changes")

        return result

    def _validate_nlp_modify_new(self, task: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and set defaults for NLP modify payload.

        Same basic defaults as _validate_nlp_modify, but uses new _enforce_fixed_time_rules.

        Input:
            task: dict with plain values (already parsed and merged)

        Returns:
            task: dict with validated values
        """
        result = task.copy()
        now = datetime.now()

        name_value = result.get("name")
        if name_value is None or name_value == "":
            result["name"] = "task"
            logger.info("Validation: name set to 'task' (was None)")
        else:
            name_value = str(name_value).strip()
            if name_value:
                name_value = name_value[0].upper() + name_value[1:]
            result["name"] = name_value
            logger.info(f"Validation: name normalized to '{name_value}'")

        # Apply fixed-time rules
        result = self._enforce_fixed_time_rules(result)

        if result.get("difficulty") is None:
            result["difficulty"] = 0.5
            logger.warning("Validation: difficulty set to 0.5 (was None)")

        if result.get("duration") is None:
            result["duration"] = 30
            logger.warning("Validation: duration set to 30 (was None)")

        category_value = result.get("category")
        if category_value is None or (isinstance(category_value, list) and len(category_value) == 0):
            result["category"] = []
            logger.warning("Validation: category set to [] (was None or empty)")
        else:
            if isinstance(category_value, list):
                category_value = [str(cat).lower().strip() for cat in category_value if cat]
            result["category"] = category_value
            logger.info(f"Validation: category normalized to {category_value}")

        location_value = result.get("location")
        if location_value is None or location_value == "":
            result["location"] = "home"
            logger.warning("Validation: location set to 'home' (was None)")
        else:
            location_value = str(location_value).lower().strip()
            result["location"] = location_value
            logger.info(f"Validation: location normalized to '{location_value}'")

        if result.get("importance") is None:
            result["importance"] = 0.5
            logger.warning("Validation: importance set to 0.5 (was None)")

        return result

    # ================================================================
    # HELPER: STRIP NOOP FIXED TIME
    # ================================================================

    def _strip_noop_fixed_time(self, changed_fields: dict) -> dict:
        """
        Remove fixed_time: False if no start/deadline is being changed.

        When the parser emits fixed_time: False without touching start or
        deadline, it's a no-op — not an intent to change temporal state.
        Stripping it preserves the existing task's fixed_time value.

        Rules:
            - fixed_time is False and start not in changed_fields
              and deadline not in changed_fields
              -> remove fixed_time
            - Otherwise: no change
        """
        if (
            changed_fields.get("fixed_time") is False
            and "start" not in changed_fields
            and "deadline" not in changed_fields
        ):
            result = {k: v for k, v in changed_fields.items() if k != "fixed_time"}
            logger.info(
                f"Stripped fixed_time=False (no start/deadline in changed_fields)"
            )
            return result
        return changed_fields

    # ================================================================
    # HELPER: PRE-PROCESS FIXED START
    # ================================================================

    def _pre_process_fixed_start(self, task: dict) -> dict:
        """
        Combine start string + fixed_start string before date parsing.

        When fixed_time=True and both start and fixed_start are string values
        (but deadline is null), the two strings are concatenated so that
        parsedatetime can resolve them together to a single datetime.

        Rules:
            - fixed_time=True, fixed_start is str, start is str, deadline is null
              -> fixed_start = f"{start} {fixed_start}", start = None
            - Otherwise: no change

        Input:
            task: flat dict (raw string values, before date parsing)

        Returns:
            task: flat dict with combined fixed_start if applicable
        """
        result = task.copy()

        if (
            result.get("fixed_time") is True
            and isinstance(result.get("fixed_start"), str) and result["fixed_start"]
            and isinstance(result.get("start"), str) and result["start"]
            and (result.get("deadline") is None or result.get("deadline") == "")
        ):
            result["fixed_start"] = f"{result['start']} {result['fixed_start']}"
            result["start"] = None
            logger.info(
                f"Pre-process fixed_start: combined start + fixed_start -> '{result['fixed_start']}'"
            )

        return result

    # ================================================================
    # HELPER: DATE PARSING
    # ================================================================

    def _date_parse(self, task: dict[str, Any], fixed_time: bool = False) -> dict[str, Any]:
        """
        Parse date strings to datetime objects.

        Input:
            task: Task data (may have raw string dates)
            fixed_time: If True, only parse fixed_start (skip start/deadline)

        Returns:
            task: Task with datetime objects for date fields
        """
        result = task.copy()
        
        # For fixed_time tasks, only parse fixed_start
        if fixed_time:
            date_fields = ["fixed_start"]
        else:
            date_fields = ["start", "deadline", "fixed_start"]

        for field in date_fields:
            value = result.get(field)
            if isinstance(value, str) and value:
                parsed, flag = self._parse_date_string(value)
                if parsed:
                    if flag == 1:
                        if field == "start":
                            parsed = parsed.replace(hour=6, minute=0, second=0, microsecond=0)
                        elif field == "deadline":
                            parsed = parsed.replace(hour=23, minute=59, second=0, microsecond=0)
                        logger.debug(f"Applied default time for {field}: {parsed}")
                    result[field] = parsed
                    logger.debug(f"Parsed {field}: '{value}' -> {parsed}")
                else:
                    if flag >= 2:
                        parsed_tomorrow = self._parse_time_only_as_tomorrow(value)
                        if parsed_tomorrow:
                            result[field] = parsed_tomorrow
                            logger.info(f"Parsed {field} (tomorrow): '{value}' -> {parsed_tomorrow}")
                        else:
                            logger.warning(f"Failed to parse {field}: '{value}'")
                            result[field] = None
                    else:
                        logger.warning(f"Failed to parse {field}: '{value}'")
                        result[field] = None

        return result

    def _parse_time_only_as_tomorrow(self, time_string: str) -> Optional[datetime]:
        """
        Parse a time-only string as tomorrow's date.

        Input:
            time_string: String like "09:00" or "3pm"

        Returns:
            datetime: Tomorrow's date with the parsed time, or None if parsing fails
        """
        try:
            cal = parsedatetime.Calendar()
            parsed, flag = cal.parse(time_string)
            if flag >= 2:
                dt = datetime(*parsed[:6])
                tomorrow = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                result = tomorrow.replace(hour=dt.hour, minute=dt.minute, second=0, microsecond=0)
                logger.debug(f"Time-only '{time_string}' parsed as tomorrow: {result}")
                return result
        except Exception as e:
            logger.error(f"Time-only parsing error: {e}")
        return None

    def _parse_date_string(self, date_string: str) -> Tuple[Optional[datetime], int]:
        """
        Parse a date string using parsedatetime with future validation.

        Returns:
            (datetime, flag): datetime object and parsedatetime flag
            - flag = 1: date only (no time specified)
            - flag >= 2: date AND time specified
        """
        try:
            cal = parsedatetime.Calendar()
            parsed, flag = cal.parse(date_string)
            if flag:
                dt = datetime(*parsed[:6])
                if dt >= datetime.now():
                    return dt, flag
                else:
                    logger.warning(
                        f"Parsed date is in past: '{date_string}' -> {dt.isoformat()}"
                    )
                    return None, flag
        except Exception as e:
            logger.error(f"Date parsing error: {e}")
        return None, 0

    # ================================================================
    # HELPER: DRAFT OPERATIONS
    # ================================================================

    def _serialize_datetime(self, obj):
        """Serialize datetime objects to ISO strings for JSON storage."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, (dict, list)):
            if isinstance(obj, dict):
                return {k: self._serialize_datetime(v) for k, v in obj.items()}
            return [self._serialize_datetime(item) for item in obj]
        return obj

    def _draft_save(
        self,
        db: Session,
        task_payload: dict[str, Any],
        match_result: dict[str, Any],
    ) -> UUID:
        """
        Save task to draft table.

        Input:
            task_payload: Enriched task data
            match_result: Task matching result

        Returns:
            draft_id: UUID of saved draft
        """
        draft_id = uuid4()

        content = {
            "task": self._serialize_datetime(task_payload),
            "match_result": {
                "associated_id": str(match_result.get("associated_id"))
                if match_result.get("associated_id")
                else None,
                "association_status": match_result.get("association_status"),
                "name_vector": match_result.get("name_vector"),
            },
        }

        draft = TaskDraft(id=draft_id, content=content)
        db.add(draft)
        db.commit()

        logger.info(f"Draft saved: {draft_id}")
        return draft_id

    def _draft_load(self, db: Session, draft_id: UUID) -> Optional[dict[str, Any]]:
        """Load task from draft table and delete it."""
        draft = db.query(TaskDraft).filter(TaskDraft.id == draft_id).first()
        if draft:
            content = draft.content
            logger.debug(f"Draft loaded: {draft_id}")

            # Delete draft after loading (memory efficiency)
            db.delete(draft)
            db.commit()
            logger.info(f"Draft deleted: {draft_id}")

            return content
        logger.warning(f"Draft not found: {draft_id}")
        return None

    def _draft_merge(
        self,
        request_task: dict[str, Any],
        draft_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Merge request task with draft (request priority).

        Request fields take precedence over draft fields.
        """
        if not draft_data:
            logger.debug("No draft data, using request only")
            return request_task

        draft_task = draft_data.get("task", {})

        merged = draft_task.copy()
        for key, value in request_task.items():
            if value is not None:
                merged[key] = value
            elif key not in merged:
                merged[key] = None

        logger.debug("Merged request with draft (request priority)")
        return merged

    # ================================================================
    # HELPER: CHANGE MERGE (for NLP modify)
    # ================================================================

    def _change_merge(
        self,
        existing_task: dict[str, Any],
        changed_fields: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Merge NLP changed fields with existing task.

        Changed fields take precedence over existing fields.
        """
        merged = existing_task.copy()
        for key, value in changed_fields.items():
            if value is not None:
                merged[key] = value

        return merged

    # ================================================================
    # HELPER: COMPUTE (urgency/value)
    # ================================================================

    def _compute(self, task: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate derived fields: urgency, value, and importance (if predicted).

        Input:
            task: Task with importance, deadline, difficulty

        Returns:
            task: Task with added urgency, value, and possibly updated importance
        """
        result = task.copy()

        importance = result.get("importance", 0.5)
        deadline = result.get("deadline")
        difficulty = result.get("difficulty", 0.5)

        if result.get("fixed_time") and result.get("fixed_start"):
            urgency_reference = result["fixed_start"]
        else:
            urgency_reference = deadline

        urgency = self._calculate_urgency(importance, urgency_reference)
        value = self._calculate_value(importance, urgency, difficulty)

        result["urgency"] = urgency
        result["value"] = value

        logger.debug(f"Computed: urgency={urgency}, value={value}")
        return result

    @staticmethod
    def _calculate_urgency(importance: float, deadline: Optional[datetime]) -> float:
        """
        Calculate urgency: min(1.0, importance * (1/days_left) * 3)

        Uses total_seconds for sub-day precision.
        """
        if not deadline:
            return 0.0

        now = datetime.now()
        days_left = (deadline - now).total_seconds() / 86400

        if days_left <= 0:
            days_left = 0.001

        urgency = min(1.0, importance * (1 / days_left) * 3)
        return round(max(0.0, urgency), 2)

    @staticmethod
    def _calculate_value(
        importance: float,
        urgency: float,
        difficulty: float,
        completion_rate: float = 1.0,
    ) -> float:
        """
        Calculate composite value.

        Formula: (importance * 0.4 + urgency * 0.4 + difficulty * 0.2) * completion_rate
        """
        raw_value = (importance * 0.4) + (urgency * 0.4) + (difficulty * 0.2)
        return round(raw_value * completion_rate, 2)

    # ============================================================================
    # OUTPUT CONVERSION - Schema output for all public methods
    # ============================================================================

    def _convert_output(
        self,
        task_dict: dict[str, Any],
        with_computed: bool = False,
        with_refs: bool = False,
    ) -> TaskPayload | TaskPayloadComputed | TaskPayloadComputedWithRefs:
        """
        Convert internal dict output to Pydantic schema.

        Args:
            task_dict: Internal task dict
            with_computed: If True, include urgency and value fields
            with_refs: If True, include internal refs (task_statistics_id, name_vector, association_status)

        Returns:
            TaskPayload, TaskPayloadComputed, or TaskPayloadComputedWithRefs
        """
        if with_refs:
            return TaskPayloadComputedWithRefs(**task_dict)
        elif with_computed:
            return TaskPayloadComputed(**task_dict)
        else:
            return TaskPayload(**task_dict)


enrichment_service = EnrichmentService()
