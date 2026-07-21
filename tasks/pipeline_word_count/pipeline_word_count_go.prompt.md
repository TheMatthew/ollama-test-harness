Write a Go program that counts words in text lines using a concurrent pipeline.

Requirements:
* Hardcode these exact 5 lines of text in the program (do not read from stdin or a file):
  * `the quick brown fox`
  * `jumps over the lazy dog`
  * `hello world`
  * `go is a concurrent language`
  * `one`
* Use one producer goroutine that sends each line into a channel.
* Use two worker goroutines that receive lines from the shared channel.
* Each worker splits the line into words (split on whitespace) and counts the number of words.
* Each worker prints one result line per input line it processes, using the exact format:
  `counted <N> words in "<line>"`
  where `<N>` is the word count and `<line>` is the original line text.
* After all lines are processed, the main goroutine prints a single summary line:
  `total lines processed: 5`
* Use channels for communication and a sync.WaitGroup to coordinate completion.
* The program must process all 5 lines exactly once and terminate cleanly.
* Keep the solution self-contained in a single Go source file.

Return only the Go source code inside a single ```go ... ``` block.
