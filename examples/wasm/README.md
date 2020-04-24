# How to build

Add the `wasm32` target to Rust, build it with that target & copy it to the root:

```bash
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/examples/*.wasm examples/wasm
```

Now we have to host the website:

```bash
cargo install basic-http-server
basic-http-server examples/wasm
```
