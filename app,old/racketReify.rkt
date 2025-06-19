#!/usr/bin/env racket
#lang racket

;; ——————————————————————————————————————————————————————————
;; Ollama client (blocking, no external deps but Racket stdlib)
;; ——————————————————————————————————————————————————————————
(require net/http-client
         racket/string
         net/url
         json
         racket/cmdline
         file/convertible)

(struct ollama-client (host port) #:transparent)

(define default-client (ollama-client "localhost" 11434))

;; Low-level POST that returns a jsexpr
(define (ollama-request client endpoint payload)
  (define host (ollama-client-host client))
  (define port (ollama-client-port client))
  (define json-bytes (string->bytes/utf-8 (jsexpr->string payload)))

  (define req-line
    (format "POST ~a HTTP/1.1\r\nHost: ~a\r\nContent-Type: application/json\r\nContent-Length: ~a\r\nConnection: close\r\n\r\n"
            endpoint host (bytes-length json-bytes)))

  (define-values (in out) (tcp-connect host port))
  (write-bytes (string->bytes/utf-8 req-line) out)
  (write-bytes json-bytes out)
  (flush-output out)

  (define resp (port->string in))  ;; grab entire response
  (close-input-port in) (close-output-port out)

  ;; Split headers/body crudely at first empty line
  (define delimiter (string-contains? resp "\r\n\r\n"))
  (unless delimiter (error "No HTTP body in response"))
  (define body (substring resp (+ delimiter 4)))

(define (ollama-generate client prompt #:model [model "gemma2:latest"])
  (define resp
    (ollama-request client "/api/generate"
                    (hash 'model model 'prompt prompt 'stream #f)))
  (cond
    [(hash-has-key? resp 'choices)
     (hash-ref (list-ref (hash-ref resp 'choices) 0) 'text "")]
    [(hash-has-key? resp 'response)
     (hash-ref resp 'response "")]
    [else ""]))

;; ——————————————————————————————————————————————————————————
;; Helpers for reading / writing *this* script file
;; ——————————————————————————————————————————————————————————
(define (self-path)
  (define argv (current-command-line-arguments))
  (if (positive? (vector-length argv))
      (vector-ref argv 0)
      (error 'self-path "Run script with `racket this-file.rkt` so argv[0] is available")))

(define (slurp path)
  (call-with-input-file path #:mode 'text port->string))

(define (spit path content)
  (call-with-output-file path #:exists 'truncate/replace
    (lambda (out) (display content out))))

;; ——————————————————————————————————————————————————————————
;; Prompt template: tell the model to *return ONLY code*
;; ——————————————————————————————————————————————————————————
(define (make-prompt code)
  (string-append
   "You are a reflexive Racket Quine engine. "
   "Return ONLY valid `#lang racket` source code that still rewrites itself. "
   "Here is the current version delimited by triple backticks.\n\n"
   "```racket\n" code "\n```\n\n"
   "Produce an improved replacement, nothing else."))

;; ——————————————————————————————————————————————————————————
;; CLI — call Ollama, overwrite file, announce result
;; ——————————————————————————————————————————————————————————
(define-values (raw-model maybe-path)
  (command-line
   #:program "racketReify.rkt"
   #:once-each [("-m" "--model") raw-model "Ollama model" "gemma2:latest"]
   #:args [maybe-path]
   (values "gemma2:latest" '()))) ; defaults if nothing at all

(define model       (string->symbol raw-model))
(define script-path (if (null? maybe-path) (self-path) (car maybe-path)))

(define current-code (slurp script-path))
(define prompt (make-prompt current-code))
(define new-code (ollama-generate default-client prompt #:model model))
(define (string-blank? s)
  (regexp-match? #px"^\\s*$" s))

(if (string-blank? new-code)
    (begin
      (displayln "Ollama returned empty text — no update performed."))
    (begin
      (spit script-path new-code)
      (displayln "Script updated successfully by Ollama!")))
)