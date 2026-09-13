import { index, sqliteTable, text } from 'drizzle-orm/sqlite-core';

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
