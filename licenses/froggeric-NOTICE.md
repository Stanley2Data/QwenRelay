# Qwen fixed chat template attribution

The bundled `config/chat-template.jinja` is based on the Qwen-family fixed template by **Frédéric Guigand (@froggeric)** and contributors, [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates). Its author declares Apache-2.0, inherited from Qwen. See the root Apache-2.0 LICENSE.

Reference public version: commit `4ea21db90694e60d002500dae85ebff26e4b23ad`, `chat_template.jinja`, SHA256 `e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2` (v22.5).

The tested variant keeps its `qwen3.8-froggeric-v22.5.0` marker and adds the concise-answer instruction / `terse` switch and runtime tool-protocol suppression. It is retained from the tested project profile, not claimed as original QwenStride template authorship. Its SHA256 is `cdff39fb26b60dc90faa292e726655c6b21f62db497846e02e4c4bbab942a84a`.

The model-author assets and original model templates have separate provenance in `locks/models.json`. This notice does not change template text or model behavior.
