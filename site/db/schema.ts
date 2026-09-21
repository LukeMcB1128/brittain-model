import { index, integer, sqliteTable, text, uniqueIndex } from 'drizzle-orm/sqlite-core';

export const users = sqliteTable('user', {
  id: text('id').primaryKey(),
  name: text('name').notNull(),
  email: text('email').notNull().unique(),
  emailVerified: integer('emailVerified', { mode: 'boolean' }).notNull(),
  image: text('image'),
  createdAt: text('createdAt').notNull(),
  updatedAt: text('updatedAt').notNull(),
});

export const sessions = sqliteTable('session', {
  id: text('id').primaryKey(),
  expiresAt: text('expiresAt').notNull(),
  token: text('token').notNull().unique(),
  createdAt: text('createdAt').notNull(),
  updatedAt: text('updatedAt').notNull(),
  ipAddress: text('ipAddress'),
  userAgent: text('userAgent'),
  userId: text('userId').notNull().references(() => users.id, { onDelete: 'cascade' }),
}, table => [index('idx_session_userId').on(table.userId)]);

export const accounts = sqliteTable('account', {
  id: text('id').primaryKey(),
  accountId: text('accountId').notNull(),
  providerId: text('providerId').notNull(),
  userId: text('userId').notNull().references(() => users.id, { onDelete: 'cascade' }),
  accessToken: text('accessToken'),
  refreshToken: text('refreshToken'),
  idToken: text('idToken'),
  accessTokenExpiresAt: text('accessTokenExpiresAt'),
  refreshTokenExpiresAt: text('refreshTokenExpiresAt'),
  scope: text('scope'),
  password: text('password'),
  createdAt: text('createdAt').notNull(),
  updatedAt: text('updatedAt').notNull(),
}, table => [
  index('idx_account_userId').on(table.userId),
  uniqueIndex('idx_account_provider_account').on(table.providerId, table.accountId),
]);

export const verifications = sqliteTable('verification', {
  id: text('id').primaryKey(),
  identifier: text('identifier').notNull(),
  value: text('value').notNull(),
  expiresAt: text('expiresAt').notNull(),
  createdAt: text('createdAt').notNull(),
  updatedAt: text('updatedAt').notNull(),
}, table => [index('idx_verification_identifier').on(table.identifier)]);

export const chats = sqliteTable('chats', {
  id: text('id').primaryKey(),
  userId: text('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  title: text('title').notNull(),
  payload: text('payload').notNull(),
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
}, table => [index('idx_chats_user_updated').on(table.userId, table.updatedAt)]);

export const chatExchanges = sqliteTable('chat_exchanges', {
  id: text('id').primaryKey(),
  createdAt: text('created_at').notNull(),
  userHash: text('user_hash').notNull(),
  model: text('model').notNull(),
  payload: text('payload').notNull(),
}, table => [
  index('idx_chat_exchanges_created_at').on(table.createdAt),
  index('idx_chat_exchanges_user_created_at').on(table.userHash, table.createdAt),
]);

// AISD/TEA course curriculum: one row per course markdown file.
//
// No full-text index on purpose. The corpus is 394 courses totalling 2.23 MB,
// which SQLite scans in milliseconds, and the tool only runs when the model
// asks for it. FTS5 would add a virtual table, shadow tables and sync triggers
// that drizzle cannot express, for no measurable gain at this size. Revisit if
// the corpus grows by an order of magnitude.
//
// `body` holds the whole file. A course averages 5.8 KB -- about 1,450 tokens --
// so a lookup can return the complete document rather than a chunk, which is
// what keeps TEKS language quotable rather than paraphrased.
export const courses = sqliteTable('courses', {
  slug: text('slug').primaryKey(),        // math/ap-calculus-ab
  subject: text('subject').notNull(),     // math
  title: text('title').notNull(),         // AP Calculus AB
  credit: text('credit'),
  gradeLevel: text('grade_level'),
  courseNumber: text('course_number'),
  peims: text('peims'),
  teksCite: text('teks_cite'),
  body: text('body').notNull(),
  updatedAt: text('updated_at').notNull(),
}, table => [
  index('idx_courses_subject').on(table.subject),
  index('idx_courses_title').on(table.title),
]);
