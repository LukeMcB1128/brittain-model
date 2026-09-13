CREATE TABLE `chat_exchanges` (
	`id` text PRIMARY KEY NOT NULL,
	`created_at` text NOT NULL,
	`user_hash` text NOT NULL,
	`model` text NOT NULL,
	`payload` text NOT NULL
);
--> statement-breakpoint
CREATE INDEX `idx_chat_exchanges_created_at` ON `chat_exchanges` (`created_at`);--> statement-breakpoint
CREATE INDEX `idx_chat_exchanges_user_created_at` ON `chat_exchanges` (`user_hash`,`created_at`);