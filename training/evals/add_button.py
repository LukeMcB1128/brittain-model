"""Add the Continue button beside Try again, for story models only."""
import pathlib

p = pathlib.Path(r"C:\Coding\brittain-model\site\src\App.jsx")
s = p.read_text(encoding="utf-8")

old = """                              {index === messages.length - 1 && (
                                <button
                                  className="retry-button"
                                  disabled={busy || !selectedModel}
                                  onClick={() => send(true)}
                                >
                                  <Icon name="retry" />
                                  Try again
                                </button>
                              )}"""
new = """                              {index === messages.length - 1 && (
                                <button
                                  className="retry-button"
                                  disabled={busy || !selectedModel}
                                  onClick={() => send(true)}
                                >
                                  <Icon name="retry" />
                                  Try again
                                </button>
                              )}
                              {/* Story models cannot be asked to carry on in
                                  words -- they were tuned on one request and
                                  one story, so a second turn reads as a new
                                  request. The button says "continue" to the
                                  server, which reframes the story so far as a
                                  longer document instead. */}
                              {index === messages.length - 1 &&
                                story &&
                                text && (
                                  <button
                                    className="retry-button"
                                    disabled={busy || !selectedModel}
                                    onClick={() => send(false, true)}
                                  >
                                    <Icon name="feather" />
                                    Continue
                                  </button>
                                )}"""
assert s.count(old) == 1
p.write_text(s.replace(old, new), encoding="utf-8")
print("button added")
