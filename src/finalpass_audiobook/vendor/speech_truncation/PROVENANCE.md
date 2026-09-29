# Vendored: speech-truncation-detection-12M

Upstream: <https://huggingface.co/mythicinfinity/speech-truncation-detection-12M>
Revision: `75ae15a354568c179c910a833129f68e3a845218`
Licence: Apache-2.0 (`LICENSE`, unmodified).

Every file here was read in full and is **unmodified** from that revision:

| file | SHA-256 |
|---|---|
| `__init__.py` | `89757e93abba6f3430ad065e7eac53b92dff657da0319836a3f2c2ecc31de055` |
| `configuration_speech_truncation_detection.py` | `2f917acfef16caded578bd752a84a168cdaeeabc6d2fe062423150ad3f1c8a9d` |
| `modeling_speech_truncation_detection.py` | `5ff191a9051c6fe8a1fe63a570a7c0837102db681274da027f9a101107b79814` |
| `processing_speech_truncation_detection.py` | `7c524b267764b32a0a1227f47cc28f9e337dbd256a95eb3b14f037de20b6dcde` |
| `config.json` | `e290248921d3b678f7940b3cc2728e89c46ef26a00d7e676d1e7e7a76d538f49` |
| `README.md` | `349de03d02bab85c37839e41dd1579e19635c0664216e70998344c2ae63508cb` |
| `LICENSE` | `58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd` |

The weights are **not** vendored. `fpab setup-model` installs `model.safetensors`
(51,820,912 bytes, SHA-256
`5e150a897364f4afdfb557584df9607f4217ea99e08b74186ffe919f67caa0c0`) and refuses
any file that does not match.

Loading never uses `trust_remote_code`: the classes are imported from this
directory, the config is built from `config.json`, and the safetensors weights
are loaded with `load_state_dict(strict=True)` — the README's own
"Non-Remote-Code Fallback".
