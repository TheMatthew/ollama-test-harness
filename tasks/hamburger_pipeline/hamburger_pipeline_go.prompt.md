Write a thread-safe Go program that models a producer-consumer pipeline for making hamburgers.

Requirements:
* Use one producer goroutine that produces exactly 8 bread, 4 hamburger patties, and 4 tomatoes.
* Use two consumer goroutines.
* Each consumer must assemble hamburgers by taking 2 bread, 1 burger patty, and 1 tomato from shared state.
* The program must make exactly 4 hamburgers total.
* The shared ingredient pool must be protected so the program is thread-safe.
* Synchronize access with Go concurrency primitives such as channels, mutexes, wait groups, or a combination of them.
* Print one line each time the producer adds an ingredient, using these exact prefixes:
  * `produced bread`
  * `produced burger`
  * `produced tomato`
* Print one line each time a consumer assembles a hamburger, using the exact prefix `assembled hamburger`.
* Avoid data races and unsafe shared-memory access.
* Keep the solution self-contained in a single Go source file.

Return only the Go source code inside a single ```go ... ``` block.
