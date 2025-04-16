# 自动向量化与基于 intrinsic 的手动向量化 小作业 实验报告 

颜子俊 2023010828

## 运行时间

```
baseline: 4437 us
auto simd: 526 us
intrinsic: 528 us
```

## 实现代码

```cpp
void a_plus_b_intrinsic(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i += 8) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vb = _mm256_load_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_store_ps(c + i, vc);
    }
}
```