"""Rewrite the three tests that asserted multi-turn history for story models."""
import pathlib
import re

p = pathlib.Path(r"C:\Coding\brittain-model\tests\test_chat_context.py")
s = p.read_text(encoding="utf-8")

s = s.replace(
    "from brittain.chat_context import frame_conversation, prepare_chat_context, validate_messages",
    "from brittain.chat_context import (\n"
    "    extend_story,\n"
    "    frame_conversation,\n"
    "    prepare_chat_context,\n"
    "    validate_messages,\n"
    ")",
)

old_followup = '''    def test_followup_includes_both_sides_and_private_story_tags(self):
        messages = [msg('user', 'Her name is Mira.'),
                    msg('assistant', '<|tags|>[Voice: Modern]<|end_tags|>Mira sailed.'),
                    msg('user', 'What is her name?')]
        for story in (False, True):
            prompt, meta = prepare_chat_context(messages, list, 4096, 256, story=story)
            self.assertIn('Her name is Mira.', prompt)
            self.assertIn('Mira sailed.', prompt)
            self.assertIn('What is her name?', prompt)
            self.assertEqual(meta['dropped_messages'], 0)
        self.assertIn('### Input:', frame_conversation(messages))
        self.assertIn('<|assistant|><|tags|>', frame_conversation(messages, story=True))
'''

new_followup = '''    def test_code_models_receive_the_conversation(self):
        messages = [msg('user', 'Her name is Mira.'),
                    msg('assistant', '<|tags|>[Voice: Modern]<|end_tags|>Mira sailed.'),
                    msg('user', 'What is her name?')]
        prompt, meta = prepare_chat_context(messages, list, 4096, 256, story=False)
        self.assertIn('Her name is Mira.', prompt)
        self.assertIn('Mira sailed.', prompt)
        self.assertIn('What is her name?', prompt)
        self.assertEqual(meta['dropped_messages'], 0)
        self.assertIn('### Input:', frame_conversation(messages))

    def test_story_models_receive_only_the_latest_request(self):
        # This test asserted the opposite, and the behaviour it locked in is why
        # a story about a president came back as more sailor. Every SFT example
        # is one request and one story, so the model has never seen a second
        # <|user|> and does not read turn structure: given a conversation it
        # ignores the newest instruction and keeps writing the previous story.
        messages = [msg('user', 'Her name is Mira.'),
                    msg('assistant', '<|tags|>[Voice: Modern]<|end_tags|>Mira sailed.'),
                    msg('user', 'Write about a president.')]
        prompt, meta = prepare_chat_context(messages, list, 4096, 256, story=True)
        self.assertEqual(
            prompt, '<|user|>Write about a president.<|end_message|><|assistant|>')
        self.assertNotIn('Mira sailed.', prompt)
        self.assertTrue(meta['single_turn'])
        self.assertFalse(meta['continued'])
        # Not a context overflow, so a client must not warn about one.
        self.assertEqual(meta['dropped_messages'], 0)

    def test_continuing_extends_the_story_instead_of_asking_again(self):
        # Carrying on is the PRETRAINING frame, not a longer conversation:
        # pretraining is almost entirely novel windows continuing from the one
        # before, and every SFT example ends at <|story_end|>.
        messages = [msg('user', 'Write about a sailor.'),
                    msg('assistant', '<|tags|>[Setting: Sea]<|end_tags|>The sea is grey.'),
                    msg('user', 'continue')]
        prompt, meta = prepare_chat_context(
            messages, list, 4096, 256, story=True, continuing=True)
        self.assertEqual(
            prompt, '<|story_start|><|tags|>[Setting: Sea]<|end_tags|>The sea is grey.')
        self.assertTrue(meta['continued'])
        self.assertFalse(meta['single_turn'])

    def test_continuing_with_nothing_to_continue_is_a_fresh_request(self):
        messages = [msg('user', 'continue')]
        prompt, meta = prepare_chat_context(
            messages, list, 4096, 256, story=True, continuing=True)
        self.assertEqual(prompt, '<|user|>continue<|end_message|><|assistant|>')
        self.assertTrue(meta['single_turn'])

    def test_a_long_story_is_trimmed_from_the_front_keeping_its_tags(self):
        # The tags are the conditioning the story was written to, so dropping
        # them to keep another paragraph would let the continuation drift out of
        # the genre it is continuing.
        story = '<|tags|>[Setting: Sea]<|end_tags|>One one one.\\n\\nTwo two two.\\n\\nThree.'
        kept = extend_story(story, encode=str.split, budget=9)
        self.assertIn('[Setting: Sea]', kept)
        self.assertIn('Three.', kept)
        self.assertNotIn('One one one.', kept)
        with self.assertRaises(ValueError):
            extend_story(story, encode=str.split, budget=2)
'''
assert s.count(old_followup) == 1
s = s.replace(old_followup, new_followup)

old_budget = """        messages = [system, msg('user', 'x' * 250), msg('assistant', 'y' * 250), latest]
        for story in (False, True):
            prompt, meta = prepare_chat_context(messages, list, 512, 64, story=story, frame_tokens=7)
            self.assertEqual(prompt, frame_conversation([system, latest], story))
            self.assertEqual(meta['dropped_messages'], 2)"""
new_budget = """        messages = [system, msg('user', 'x' * 250), msg('assistant', 'y' * 250), latest]
        # Story models are single-turn regardless of budget, so turn-dropping is
        # only meaningful for the models that receive a conversation at all.
        for story in (False,):
            prompt, meta = prepare_chat_context(messages, list, 512, 64, story=story, frame_tokens=7)
            self.assertEqual(prompt, frame_conversation([system, latest], story))
            self.assertEqual(meta['dropped_messages'], 2)"""
assert s.count(old_budget) == 1
s = s.replace(old_budget, new_budget)

old_endpoint = """            self.assertEqual(final['context']['dropped_messages'], 0)
            self.assertIn('Remember Mira.', self.prompts[-1][0])
            self.assertIn('Mira is here.', self.prompts[-1][0])"""
new_endpoint = """            self.assertEqual(final['context']['dropped_messages'], 0)
            # A story model is framed as one turn, so the newest request is the
            # whole prompt and the earlier turns are absent by design.
            self.assertIn('Continue.', self.prompts[-1][0])
            self.assertNotIn('Mira is here.', self.prompts[-1][0])
            self.assertTrue(final['context']['single_turn'])"""
assert s.count(old_endpoint) == 1
s = s.replace(old_endpoint, new_endpoint)

s = s.replace("async def test_stream_and_nonstream_include_history_and_context(self):",
              "async def test_stream_and_nonstream_frame_one_turn_and_report_context(self):")

p.write_text(s, encoding="utf-8")
print("tests updated")
