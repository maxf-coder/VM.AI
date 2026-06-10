-- ============================================================================
-- VM.AI Database - Show All Records Across All Tables
-- ============================================================================

-- 1. Tasks (main)
SELECT 'tasks' AS table_name, COUNT(*) AS count FROM tasks;
SELECT * FROM tasks ORDER BY created_at DESC LIMIT 10;

-- 2. Categories
SELECT 'categories' AS table_name, COUNT(*) AS count FROM categories;
SELECT * FROM categories;

-- 3. Locations
SELECT 'locations' AS table_name, COUNT(*) AS count FROM locations;
SELECT * FROM locations;

-- 4. Task-Category Junction
SELECT 'task_categories' AS table_name, COUNT(*) AS count FROM task_categories;
SELECT tc.*, t.name AS task_name, c.name AS category_name 
FROM task_categories tc
JOIN tasks t ON tc.task_id = t.id
JOIN categories c ON tc.category_id = c.id;

-- 5. Task Drafts
SELECT 'task_drafts' AS table_name, COUNT(*) AS count FROM task_drafts;
SELECT * FROM task_drafts ORDER BY created_at DESC;

-- 6. Task Statistics
SELECT 'task_statistics' AS table_name, COUNT(*) AS count FROM task_statistics;
SELECT * FROM task_statistics;

-- 7. Category Statistics
SELECT 'category_statistics' AS table_name, COUNT(*) AS count FROM category_statistics;
SELECT cs.*, c.name AS category_name 
FROM category_statistics cs
JOIN categories c ON cs.category_id = c.id;

-- 8. Task Statistics Locations
SELECT 'task_statistics_locations' AS table_name, COUNT(*) AS count FROM task_statistics_locations;
SELECT tsl.*, ts.task_name, l.name AS location_name
FROM task_statistics_locations tsl
JOIN task_statistics ts ON tsl.statistics_id = ts.id
JOIN locations l ON tsl.location_id = l.id;

-- 9. Category Statistics Locations
SELECT 'category_statistics_locations' AS table_name, COUNT(*) AS count FROM category_statistics_locations;
SELECT csl.*, c.name AS category_name, l.name AS location_name
FROM category_statistics_locations csl
JOIN category_statistics cs ON csl.statistics_id = cs.id
JOIN categories c ON cs.category_id = c.id
JOIN locations l ON csl.location_id = l.id;

-- 10. Main Schedule
SELECT 'main_schedule' AS table_name, COUNT(*) AS count FROM main_schedule;
SELECT * FROM main_schedule ORDER BY start;

-- 11. Provisional Schedule
SELECT 'provisional_schedule' AS table_name, COUNT(*) AS count FROM provisional_schedule;
SELECT * FROM provisional_schedule ORDER BY start;

-- 12. Unscheduled Tasks
SELECT 'unscheduled_tasks' AS table_name, COUNT(*) AS count FROM unscheduled_tasks;
SELECT ut.*, t.name AS task_name
FROM unscheduled_tasks ut
JOIN tasks t ON ut.task_id = t.id
ORDER BY ut.created_at;

-- 13. Schedule Changes Log
SELECT 'schedule_changes' AS table_name, COUNT(*) AS count FROM schedule_changes;
SELECT * FROM schedule_changes ORDER BY created_at DESC;

-- Summary counts
SELECT 
    (SELECT COUNT(*) FROM tasks) AS tasks_count,
    (SELECT COUNT(*) FROM categories) AS categories_count,
    (SELECT COUNT(*) FROM locations) AS locations_count,
    (SELECT COUNT(*) FROM task_drafts) AS drafts_count,
    (SELECT COUNT(*) FROM task_statistics) AS task_stats_count,
    (SELECT COUNT(*) FROM category_statistics) AS cat_stats_count,
    (SELECT COUNT(*) FROM main_schedule) AS main_schedule_count,
    (SELECT COUNT(*) FROM unscheduled_tasks) AS unscheduled_count;