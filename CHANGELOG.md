# CHANGELOG



## v0.4.0 (2025-01-10)

### Feature

* feat: add convenience functions for working with one layer ([`4c24c8a`](https://github.com/kalekundert/torchyield/commit/4c24c8a90b33ac12ef2b670efdb632c77f63744b))

* feat: add more nonlinearities ([`f997ab2`](https://github.com/kalekundert/torchyield/commit/f997ab2ec810188f62574d5814c814836c6dd007))


## v0.3.0 (2024-11-08)

### Feature

* feat: provide an immutable sequential module ([`2c8bcad`](https://github.com/kalekundert/torchyield/commit/2c8bcad1e6d09e52c0336abebe706971317eed1d))

* feat: monkeypatch Tensor repr to only show shape ([`63aa908`](https://github.com/kalekundert/torchyield/commit/63aa908726dd72c689175598b76b10c919a5a2f8))

### Fix

* fix: interpret an empty layer iterable as an identity module ([`3cb36a5`](https://github.com/kalekundert/torchyield/commit/3cb36a5d337bad2cf998716b91edcb7cec11bc5f))


## v0.2.0 (2024-09-17)

### Chore

* chore: remove TODO file ([`722f7eb`](https://github.com/kalekundert/torchyield/commit/722f7eb8043ada3b6ad6daf8d7de5ab614c3596c))

### Feature

* feat: don&#39;t unnecessarily wrap modules

Instead of providing a `Layers` class, provide a `module_from_layers()`.
If all the layers put together produce only a single module, this makes
it possible to simply return that module, without any wrapping. ([`7e6fdc1`](https://github.com/kalekundert/torchyield/commit/7e6fdc1596423d974267dc47c53d350c40104fb1))

* feat: allow verbose modules to take multiple arguments ([`0579427`](https://github.com/kalekundert/torchyield/commit/05794276b7e8a555e81db00bcf8fe42a183b4118))


## v0.1.0 (2024-06-05)

### Chore

* chore: recommit release workflow ([`4cdb711`](https://github.com/kalekundert/torchyield/commit/4cdb711c6f974e4567bba551736701872df42d17))

* chore: apply cookiecutter ([`abe4f57`](https://github.com/kalekundert/torchyield/commit/abe4f57b742d14a1445ad81e623f70fb3fc7d128))

### Documentation

* docs: explain why this library is useful ([`f208c68`](https://github.com/kalekundert/torchyield/commit/f208c68522fe917701be7f8446210668f42be542))

### Feature

* feat: support pooling in dynamic factories ([`d639d0b`](https://github.com/kalekundert/torchyield/commit/d639d0be893aa836da68b17e0f5369b9ed987b8f))

* feat: add dynamically-generated layer factories ([`0052046`](https://github.com/kalekundert/torchyield/commit/005204632e36ac2b9487109ed2f15844ab85688e))

* feat: rename project to &#39;torchyield&#39; ([`e7a1898`](https://github.com/kalekundert/torchyield/commit/e7a1898085062758c2eeb00bae87e9ba07562730))

* feat: skip ReLU after last layer in MLP ([`bb70ce2`](https://github.com/kalekundert/torchyield/commit/bb70ce2d3bd3dccf7db4cd85accf29ef1126b809))

* feat: initial commit ([`38a7a77`](https://github.com/kalekundert/torchyield/commit/38a7a77dd33d538863840b002b391dfc892efd12))

### Fix

* fix: restore the  option ([`f3a2581`](https://github.com/kalekundert/torchyield/commit/f3a25817b1b95f0ed33fd6f8f8e14ef9be3587c7))
