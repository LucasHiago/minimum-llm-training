"""
prepare_cpp.py — Monta um corpus de C++ a partir dos seus arquivos
==================================================================

O corpus de exemplo (data/cpp.txt) e pequeno, so pra testar. Para o modelo
gerar C++ de verdade, ele precisa ver MUITO codigo. Este script varre uma
pasta, junta todos os arquivos C/C++ num unico arquivo de treino e mostra
algumas estatisticas.

Uso:

    # varre ./data/cpp_src e salva em data/cpp.txt
    python prepare_cpp.py

    # aponta para os seus projetos
    python prepare_cpp.py --src ~/meus_projetos --out data/cpp.txt

    # depois, treine:
    python train.py --data data/cpp.txt --iters 5000

Dica: quanto mais consistente o estilo do codigo, melhor o resultado. Misturar
muitos estilos diferentes deixa o aprendizado mais dificil para um modelo pequeno.
"""

import argparse
import os

# Extensoes consideradas "codigo C/C++".
EXTS = {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hh", ".hxx", ".ipp"}

# Pastas que quase nunca contem codigo-fonte util (evita lixo no corpus).
SKIP_DIRS = {"build", "cmake-build-debug", "cmake-build-release",
             ".git", "node_modules", "third_party", "vendor", "out"}


def coletar(src_dir):
    """Retorna a lista de caminhos de arquivos C/C++ encontrados."""
    arquivos = []
    for raiz, dirs, nomes in os.walk(src_dir):
        # Poda pastas indesejadas in-place (mais rapido que filtrar depois).
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for nome in nomes:
            if os.path.splitext(nome)[1].lower() in EXTS:
                arquivos.append(os.path.join(raiz, nome))
    return sorted(arquivos)


def main():
    p = argparse.ArgumentParser(description="Monta um corpus de C++ para treino.")
    p.add_argument("--src", default="data/cpp_src",
                   help="pasta a varrer em busca de arquivos C/C++")
    p.add_argument("--out", default="data/cpp.txt", help="arquivo de saida")
    p.add_argument("--min_bytes", type=int, default=0,
                   help="ignora arquivos menores que isto (bytes)")
    args = p.parse_args()

    if not os.path.isdir(args.src):
        raise SystemExit(
            f"Pasta '{args.src}' nao existe.\n"
            f"Crie-a e coloque seus arquivos C++ dentro, ou use --src <pasta>."
        )

    arquivos = coletar(args.src)
    if not arquivos:
        raise SystemExit(f"Nenhum arquivo C/C++ encontrado em '{args.src}'.")

    total_chars = 0
    usados = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as saida:
        for caminho in arquivos:
            try:
                with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
                    conteudo = f.read()
            except OSError:
                continue
            if len(conteudo.encode("utf-8")) < args.min_bytes:
                continue
            # Separador comentado marca o inicio de cada arquivo.
            rel = os.path.relpath(caminho, args.src)
            saida.write(f"// ===== {rel} =====\n")
            saida.write(conteudo)
            if not conteudo.endswith("\n"):
                saida.write("\n")
            saida.write("\n")
            total_chars += len(conteudo)
            usados += 1

    print(f"Arquivos usados: {usados} de {len(arquivos)} encontrados")
    print(f"Corpus salvo em: {args.out} ({total_chars:,} caracteres)")
    print(f"Treine com:  python train.py --data {args.out} --iters 5000")


if __name__ == "__main__":
    main()
