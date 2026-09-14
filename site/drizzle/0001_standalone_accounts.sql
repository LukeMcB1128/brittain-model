CREATE TABLE `user` (
  `id` text PRIMARY KEY NOT NULL,
  `name` text NOT NULL,
  `email` text NOT NULL UNIQUE,
  `emailVerified` integer NOT NULL,
  `image` text,
  `createdAt` date NOT NULL,
  `updatedAt` date NOT NULL
);
--> statement-breakpoint
CREATE TABLE `session` (
  `id` text PRIMARY KEY NOT NULL,
  `expiresAt` date NOT NULL,
  `token` text NOT NULL UNIQUE,
  `createdAt` date NOT NULL,
  `updatedAt` date NOT NULL,
  `ipAddress` text,
  `userAgent` text,
  `userId` text NOT NULL REFERENCES `user`(`id`) ON DELETE CASCADE
);
--> statement-breakpoint
CREATE INDEX `idx_session_userId` ON `session` (`userId`);
--> statement-breakpoint
CREATE TABLE `account` (
  `id` text PRIMARY KEY NOT NULL,
  `accountId` text NOT NULL,
  `providerId` text NOT NULL,
  `userId` text NOT NULL REFERENCES `user`(`id`) ON DELETE CASCADE,
  `accessToken` text,
  `refreshToken` text,
  `idToken` text,
  `accessTokenExpiresAt` date,
  `refreshTokenExpiresAt` date,
  `scope` text,
  `password` text,
  `createdAt` date NOT NULL,
  `updatedAt` date NOT NULL
);
--> statement-breakpoint
CREATE INDEX `idx_account_userId` ON `account` (`userId`);
--> statement-breakpoint
CREATE UNIQUE INDEX `idx_account_provider_account` ON `account` (`providerId`, `accountId`);
--> statement-breakpoint
CREATE TABLE `verification` (
  `id` text PRIMARY KEY NOT NULL,
  `identifier` text NOT NULL,
  `value` text NOT NULL,
  `expiresAt` date NOT NULL,
  `createdAt` date NOT NULL,
  `updatedAt` date NOT NULL
);
--> statement-breakpoint
CREATE INDEX `idx_verification_identifier` ON `verification` (`identifier`);
--> statement-breakpoint
CREATE TABLE `chats` (
  `id` text PRIMARY KEY NOT NULL,
  `user_id` text NOT NULL REFERENCES `user`(`id`) ON DELETE CASCADE,
  `title` text NOT NULL,
  `payload` text NOT NULL,
  `created_at` text NOT NULL,
  `updated_at` text NOT NULL
);
--> statement-breakpoint
CREATE INDEX `idx_chats_user_updated` ON `chats` (`user_id`, `updated_at`);
