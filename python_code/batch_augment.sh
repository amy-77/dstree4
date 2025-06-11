#!/bin/bash

# 批量数据增强脚本
# 使用方法: ./batch_augment.sh [输入目录] [输出目录] [可选参数]

set -e  # 遇到错误立即退出

# 默认参数
INPUT_DIR=""
OUTPUT_DIR=""
METHOD="exponential"
FACTOR=2.0
COVERAGE_STEP=0.001
MAX_ERROR_AT_COV1=10.0
ADD_EXTREMES=""
VISUALIZE=""

# 显示帮助信息
show_help() {
    echo "批量数据增强脚本"
    echo ""
    echo "使用方法:"
    echo "  $0 [输入目录] [输出目录] [选项]"
    echo ""
    echo "参数:"
    echo "  输入目录        包含CSV文件的目录"
    echo "  输出目录        增强后文件的保存目录"
    echo ""
    echo "选项:"
    echo "  --method METHOD         增强方法 (linear|exponential|quadratic), 默认: exponential"
    echo "  --factor FACTOR         增强因子, 默认: 2.0"
    echo "  --coverage-step STEP    Coverage步长, 默认: 0.001"
    echo "  --max-error VALUE       Coverage=1.0时的最大error值, 默认: 10.0"
    echo "  --add-extremes          添加极值哨兵点"
    echo "  --visualize             生成可视化图"
    echo "  --pattern PATTERN       文件名匹配模式, 默认: '*.csv'"
    echo "  --exclude PATTERN       排除的文件名模式"
    echo "  --dry-run               只显示将要处理的文件，不实际执行"
    echo "  -h, --help              显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 ./input_dir ./output_dir"
    echo "  $0 ./input_dir ./output_dir --method exponential --factor 3.0 --add-extremes"
    echo "  $0 ./input_dir ./output_dir --pattern '*_raw_data.csv' --exclude '*test*'"
}

# 解析命令行参数
PATTERN="*.csv"
EXCLUDE_PATTERN=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --factor)
            FACTOR="$2"
            shift 2
            ;;
        --coverage-step)
            COVERAGE_STEP="$2"
            shift 2
            ;;
        --max-error)
            MAX_ERROR_AT_COV1="$2"
            shift 2
            ;;
        --add-extremes)
            ADD_EXTREMES="--add_extremes"
            shift
            ;;
        --visualize)
            VISUALIZE="--visualize"
            shift
            ;;
        --pattern)
            PATTERN="$2"
            shift 2
            ;;
        --exclude)
            EXCLUDE_PATTERN="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -*)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
        *)
            if [[ -z "$INPUT_DIR" ]]; then
                INPUT_DIR="$1"
            elif [[ -z "$OUTPUT_DIR" ]]; then
                OUTPUT_DIR="$1"
            else
                echo "错误: 过多的位置参数: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# 检查必需参数
if [[ -z "$INPUT_DIR" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    echo "错误: 必须指定输入目录和输出目录"
    echo ""
    show_help
    exit 1
fi

# 检查输入目录是否存在
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 查找CSV文件
echo "搜索CSV文件..."
echo "输入目录: $INPUT_DIR"
echo "文件模式: $PATTERN"
if [[ -n "$EXCLUDE_PATTERN" ]]; then
    echo "排除模式: $EXCLUDE_PATTERN"
fi

# 使用find命令查找文件
mapfile -t csv_files < <(find "$INPUT_DIR" -name "$PATTERN" -type f)

# 过滤排除的文件
if [[ -n "$EXCLUDE_PATTERN" ]]; then
    filtered_files=()
    for file in "${csv_files[@]}"; do
        if [[ ! "$(basename "$file")" =~ $EXCLUDE_PATTERN ]]; then
            filtered_files+=("$file")
        fi
    done
    csv_files=("${filtered_files[@]}")
fi

# 检查是否找到文件
if [[ ${#csv_files[@]} -eq 0 ]]; then
    echo "警告: 在 $INPUT_DIR 中没有找到匹配的CSV文件"
    exit 0
fi

echo "找到 ${#csv_files[@]} 个CSV文件:"
for file in "${csv_files[@]}"; do
    echo "  $(basename "$file")"
done

# 如果是dry-run，只显示文件列表
if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo "这是干跑模式，实际命令将是:"
    for file in "${csv_files[@]}"; do
        filename=$(basename "$file")
        output_file="$OUTPUT_DIR/${filename%.*}_augmented.csv"
        echo "python3 data_augmentation.py --input \"$file\" --output \"$output_file\" --method $METHOD --factor $FACTOR --coverage_step $COVERAGE_STEP --max_error_at_cov1 $MAX_ERROR_AT_COV1 $ADD_EXTREMES $VISUALIZE"
    done
    exit 0
fi

# 开始批量处理
echo ""
echo "开始批量数据增强..."
echo "增强方法: $METHOD"
echo "增强因子: $FACTOR"
echo "Coverage步长: $COVERAGE_STEP"
echo "最大Error值: $MAX_ERROR_AT_COV1"
echo ""

# 统计变量
processed_count=0
failed_count=0
failed_files=()

# 处理每个文件
for file in "${csv_files[@]}"; do
    filename=$(basename "$file")
    echo "处理文件 [$((processed_count + 1))/${#csv_files[@]}]: $filename"
    
    # 构造输出文件名
    output_file="$OUTPUT_DIR/${filename%.*}_augmented.csv"
    
    # 构造命令
    cmd="python3 data_augmentation.py --input \"$file\" --output \"$output_file\" --method $METHOD --factor $FACTOR --coverage_step $COVERAGE_STEP --max_error_at_cov1 $MAX_ERROR_AT_COV1 $ADD_EXTREMES $VISUALIZE"
    
    # 执行命令
    if eval $cmd; then
        echo "✓ 成功处理: $filename -> $(basename "$output_file")"
        ((processed_count++))
    else
        echo "✗ 处理失败: $filename"
        ((failed_count++))
        failed_files+=("$filename")
    fi
    echo ""
done

# 输出统计结果
echo "==================== 批量处理完成 ===================="
echo "总文件数: ${#csv_files[@]}"
echo "成功处理: $processed_count"
echo "处理失败: $failed_count"

if [[ $failed_count -gt 0 ]]; then
    echo ""
    echo "失败文件列表:"
    for failed_file in "${failed_files[@]}"; do
        echo "  $failed_file"
    done
fi

echo ""
echo "输出目录: $OUTPUT_DIR"
echo "处理完成!" 