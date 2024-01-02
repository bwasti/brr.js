## `brr.js`

Trying to make it easier to use WebGPU for compute in JavaScript.
Super early days.
Tested on Safari Technology Preivew and Chrome Canary.

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
  const delta = ${ctx.threads};
  var i = global_invocation_index;
  for (; i < ${N}; i += delta) {
    ${ctx.args[1]}[i] = ${ctx.args[0]}[i] * 2;
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

Benchmarking:

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
