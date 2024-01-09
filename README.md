## `brr.js`
![cpuvgpu](https://github.com/bwasti/brr.js/assets/4842908/17e976c3-66f8-4dbf-83df-cbc0223bda4b) 

It's early days, but I'm trying to make it easier to use WebGPU for compute in JavaScript.
Single file (`brr.js`) tested on Safari Technology Preivew and Chrome Canary. [Try it!](https://bwasti.github.io/brr.js/)


## Usage

Simple full example:

```javascript
const brr = await import('./brr.js');
const gpu = new brr.GPU();

const N = 1024 * 1024;

// Compile an implementation by specifying input types and function body
const fn = await gpu.compile(
  [{name: 'A', type: 'vec4<f32>'},
   {name: 'B', type: 'vec4<f32>'}],
(ctx) => `
  for (var i = global_invocation_index; i < ${N}; i += ${ctx.threads}) {
    B[i] = A[i] * 2;
  }
`);

const A = gpu.alloc(Float32Array, N);
const B = gpu.alloc(Float32Array, N);

// Populate the input (A)
await A.write((arr) => {
  for (let i = 0; i < N; ++i) {
    arr[i] = i;
  }
});

// Run the function
await fn(A, B);
// Print out the underlying code
console.log(fn.code);

// Print the output (B)
await B.read((arr) => {
  console.log(arr);
});
```

Benchmarking

```javascript
// Benchmark the execution (gpu.block() only necessary for the benchmark)
const t0 = performance.now();
for (let i = 0; i < 10; ++i) {
  await fn(A, B);
}
await gpu.block();
const t1 = performance.now();

console.log(`GPU: ${Math.round(t1 - t0) / 10}us per iter`);
```

### TODO/Contributing

Feel free to submit PRs!  I'm trying to keep the basic functionality single file and relatively small.

- [ ] Simple iterators (pointwise with indexing)
- [ ] Library of common linear algebra operations (could be a different file)
- [ ] Better exceptions/checks/errors
- [ ] Improved performance for memory binding (skip copies etc)
- [ ] Automatic recompilation (and caching mechanism) for changed compile-time arguments
