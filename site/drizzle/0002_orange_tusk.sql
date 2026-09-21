CREATE TABLE `courses` (
	`slug` text PRIMARY KEY NOT NULL,
	`subject` text NOT NULL,
	`title` text NOT NULL,
	`credit` text,
	`grade_level` text,
	`course_number` text,
	`peims` text,
	`teks_cite` text,
	`body` text NOT NULL,
	`updated_at` text NOT NULL
);
--> statement-breakpoint
CREATE INDEX `idx_courses_subject` ON `courses` (`subject`);--> statement-breakpoint
CREATE INDEX `idx_courses_title` ON `courses` (`title`);