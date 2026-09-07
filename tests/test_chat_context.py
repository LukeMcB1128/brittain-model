"""CPU-only regression tests: no checkpoints, torch, or GPU required."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from brittain.chat_context import (
    extend_story,
    frame_conversation,
    prepare_chat_context,
    validate_messages,
)
from brittain.prompts import format_prompt


def msg(role, content):
    return {'role': role, 'content': content}


class ChatContextTests(unittest.TestCase):
    def test_single_turn_preserves_training_templates(self):
        messages = [msg('user', 'Write a story.')]
        self.assertEqual(frame_conversation(messages), format_prompt('Write a story.'))
        self.assertEqual(frame_conversation(messages, story=True),
                         '<|user|>Write a story.<|end_message|><|assistant|>')

    def test_code_models_receive_the_conversation(self):
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
        story = '<|tags|>[Setting: Sea]<|end_tags|>One one one.\n\nTwo two two.\n\nThree.'
        kept = extend_story(story, encode=str.split, budget=9)
        self.assertIn('[Setting: Sea]', kept)
        self.assertIn('Three.', kept)
        self.assertNotIn('One one one.', kept)
        with self.assertRaises(ValueError):
            extend_story(story, encode=str.split, budget=2)

    def test_budget_drops_whole_turns_and_keeps_system(self):
        system = msg('system', 'Be concise.')
        latest = msg('user', 'Continue.')
        messages = [system, msg('user', 'x' * 250), msg('assistant', 'y' * 250), latest]
        # Story models are single-turn regardless of budget, so turn-dropping is
        # only meaningful for the models that receive a conversation at all.
        for story in (False,):
            prompt, meta = prepare_chat_context(messages, list, 512, 64, story=story, frame_tokens=7)
            self.assertEqual(prompt, frame_conversation([system, latest], story))
            self.assertEqual(meta['dropped_messages'], 2)
            self.assertLessEqual(meta['prompt_tokens'] + meta['reserved_tokens'], 512)
            self.assertEqual(meta['prompt_tokens'], len(prompt) + 7)

    def test_oversize_latest_is_rejected_not_sliced(self):
        with self.assertRaisesRegex(ValueError, 'too long'):
            prepare_chat_context([msg('user', 'x' * 1000)], list, 512, 64)

    def test_malformed_and_excessive_requests(self):
        for bad in (None, [], {}, ['text'], [msg('tool', 'x')], [msg('user', [])], [msg('assistant', 'x')], [msg('user', '   ')], [msg('user', 'x' * 20001)]):
            with self.subTest(bad=str(bad)[:30]), self.assertRaises(ValueError):
                validate_messages(bad)

    def test_exact_budget_boundary(self):
        messages = [msg('user', 'Hello')]
        length = len(frame_conversation(messages, True))
        _, meta = prepare_chat_context(messages, list, length + 32, 32, story=True)
        self.assertEqual(meta['prompt_tokens'] + meta['reserved_tokens'], length + 32)
        with self.assertRaises(ValueError):
            prepare_chat_context(messages, list, length + 31, 32, story=True)



class ChatEndpointTests(unittest.IsolatedAsyncioTestCase):
    """Exercise the real route with a fake model, without running GPU startup."""

    def setUp(self):
        import ast
        import asyncio
        import json
        from types import SimpleNamespace
        from brittain.chat_context import PromptTooLongError
        source = Path(__file__).resolve().parents[1] / 'scripts/inference/serve.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        route = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'chat')
        route.decorator_list = []
        self.prompts = []
        self.model = SimpleNamespace(name='test', raw_default=False, story=True,
                         block=4096, max_new_tokens=512, frame_ids=[], supports_fim=False,
                                     enc=SimpleNamespace(encode=list, name='test'))

        class Response:
            def __init__(self, content, status_code=200, **kwargs):
                self.content = content
                self.status_code = status_code

        def pieces(model, prompt, raw, opts):
            self.prompts.append((prompt, raw, opts))
            yield 'A reply.'

        namespace = dict(Request=object, too_many=lambda req: None,
                         pick=lambda body: self.model, JSONResponse=Response,
                         StreamingResponse=Response, MAX_PROMPT_CHARS=20000,
                         MAX_NEW_TOKENS=512, validate_messages=validate_messages,
                         prepare_chat_context=prepare_chat_context,
                         PromptTooLongError=PromptTooLongError, stream_pieces=pieces,
                         strip_fim=lambda prompt: (prompt, None), json=json,
                         asyncio=asyncio, now=lambda: 'test-time')
        exec(compile(ast.Module(body=[route], type_ignores=[]), str(source), 'exec'), namespace)
        self.route = namespace['chat']

    async def call(self, body):
        class Request:
            async def json(self):
                return body
        return await self.route(Request())

    async def test_stream_and_nonstream_frame_one_turn_and_report_context(self):
        import json
        messages = [msg('user', 'Remember Mira.'), msg('assistant', 'Mira is here.'), msg('user', 'Continue.')]
        for stream in (True, False):
            result = await self.call({'messages': messages, 'stream': stream})
            if stream:
                chunks = [json.loads(line) for line in result.content]
                self.assertEqual(chunks[0]['message']['content'], 'A reply.')
                final = chunks[-1]
            else:
                final = result.content
                self.assertEqual(final['message']['content'], 'A reply.')
            self.assertTrue(final['done'])
            self.assertEqual(final['context']['dropped_messages'], 0)
            # A story model is framed as one turn, so the newest request is the
            # whole prompt and the earlier turns are absent by design.
            self.assertIn('Continue.', self.prompts[-1][0])
            self.assertNotIn('Mira is here.', self.prompts[-1][0])
            self.assertTrue(final['context']['single_turn'])

    async def test_raw_remains_untemplated_latest_code(self):
        self.model.raw_default = True
        result = await self.call({'messages': [msg('user', 'old'), msg('assistant', 'old reply'), msg('user', 'def add(a, b):\n    ')], 'stream': False})
        self.assertEqual(self.prompts[-1][0], 'def add(a, b):\n    ')
        self.assertTrue(self.prompts[-1][1])
        self.assertIsNone(result.content['context'])

    async def test_chat_caps_num_predict_to_selected_model(self):
        self.model.max_new_tokens = 32
        result = await self.call({
            'messages': [msg('user', 'Write a short scene.')],
            'options': {'num_predict': 500},
            'stream': False,
        })
        self.assertEqual(self.prompts[-1][2]['num_predict'], 32)
        self.assertEqual(result.content['context']['reserved_tokens'], 32)

    async def test_invalid_requests_return_errors_without_generation(self):
        for body, status in [(None, 400), ({'messages': []}, 400),
                             ({'messages': [msg('user', 'x' * 20001)]}, 413),
                             ({'messages': [msg('user', 'x')], 'options': {'num_predict': 'bad'}}, 400),
                             ({'messages': [msg('user', 'x' * 5000)]}, 400)]:
            result = await self.call(body)
            self.assertEqual(result.status_code, status)
            self.assertIn('error', result.content)
        self.assertEqual(self.prompts, [])


if __name__ == '__main__':
    unittest.main()
