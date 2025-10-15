#!/bin/bash
# Verification script for repository cleanup

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                              ║"
echo "║                    REPOSITORY CLEANUP VERIFICATION                           ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Root directory should only have tmr.py
echo "┌──────────────────────────────────────────────────────────────────────────────┐"
echo "│ TEST 1: Root Directory Python Files                                          │"
echo "└──────────────────────────────────────────────────────────────────────────────┘"
echo ""

PY_FILES=$(ls -1 *.py 2>/dev/null | wc -l)
if [ "$PY_FILES" -eq 1 ] && [ -f "tmr.py" ]; then
    echo -e "${GREEN}✅ PASS${NC}: Only tmr.py in root directory"
else
    echo -e "${RED}❌ FAIL${NC}: Expected only tmr.py, found $PY_FILES files"
    ls -1 *.py 2>/dev/null
fi
echo ""

# Test 2: Moved files should exist in new locations
echo "┌──────────────────────────────────────────────────────────────────────────────┐"
echo "│ TEST 2: Moved Files Exist                                                    │"
echo "└──────────────────────────────────────────────────────────────────────────────┘"
echo ""

FILES_OK=true

if [ -f "src/evaluation/eval_anonymization_v2.py" ]; then
    echo -e "${GREEN}✅ PASS${NC}: src/evaluation/eval_anonymization_v2.py exists"
else
    echo -e "${RED}❌ FAIL${NC}: src/evaluation/eval_anonymization_v2.py not found"
    FILES_OK=false
fi

if [ -f "src/evaluation/eval_model_main.py" ]; then
    echo -e "${GREEN}✅ PASS${NC}: src/evaluation/eval_model_main.py exists"
else
    echo -e "${RED}❌ FAIL${NC}: src/evaluation/eval_model_main.py not found"
    FILES_OK=false
fi

if [ -f "scripts/quick_test_tmr.py" ]; then
    echo -e "${GREEN}✅ PASS${NC}: scripts/quick_test_tmr.py exists"
else
    echo -e "${RED}❌ FAIL${NC}: scripts/quick_test_tmr.py not found"
    FILES_OK=false
fi

if [ -f "scripts/eval_same_action.py" ]; then
    echo -e "${GREEN}✅ PASS${NC}: scripts/eval_same_action.py exists"
else
    echo -e "${RED}❌ FAIL${NC}: scripts/eval_same_action.py not found"
    FILES_OK=false
fi
echo ""

# Test 3: Old files should not exist
echo "┌──────────────────────────────────────────────────────────────────────────────┐"
echo "│ TEST 3: Old Files Removed                                                    │"
echo "└──────────────────────────────────────────────────────────────────────────────┘"
echo ""

if [ ! -f "data.py" ]; then
    echo -e "${GREEN}✅ PASS${NC}: data.py removed"
else
    echo -e "${RED}❌ FAIL${NC}: data.py still exists"
    FILES_OK=false
fi

if [ ! -f "eval_anonymization_v2.py" ]; then
    echo -e "${GREEN}✅ PASS${NC}: eval_anonymization_v2.py removed from root"
else
    echo -e "${RED}❌ FAIL${NC}: eval_anonymization_v2.py still in root"
    FILES_OK=false
fi

if [ ! -f "eval_model.py" ]; then
    echo -e "${GREEN}✅ PASS${NC}: eval_model.py removed from root"
else
    echo -e "${RED}❌ FAIL${NC}: eval_model.py still in root"
    FILES_OK=false
fi

if [ ! -f "quick_test_tmr.py" ]; then
    echo -e "${GREEN}✅ PASS${NC}: quick_test_tmr.py removed from root"
else
    echo -e "${RED}❌ FAIL${NC}: quick_test_tmr.py still in root"
    FILES_OK=false
fi
echo ""

# Test 4: README.md should be updated
echo "┌──────────────────────────────────────────────────────────────────────────────┐"
echo "│ TEST 4: README.md Updated                                                    │"
echo "└──────────────────────────────────────────────────────────────────────────────┘"
echo ""

README_LINES=$(wc -l < README.md)
if [ "$README_LINES" -gt 600 ]; then
    echo -e "${GREEN}✅ PASS${NC}: README.md has $README_LINES lines (expected >600)"
else
    echo -e "${RED}❌ FAIL${NC}: README.md has only $README_LINES lines (expected >600)"
fi
echo ""

# Test 5: Python files compile
echo "┌──────────────────────────────────────────────────────────────────────────────┐"
echo "│ TEST 5: Python Files Compile                                                 │"
echo "└──────────────────────────────────────────────────────────────────────────────┘"
echo ""

COMPILE_OK=true

if python -m py_compile tmr.py 2>/dev/null; then
    echo -e "${GREEN}✅ PASS${NC}: tmr.py compiles"
else
    echo -e "${RED}❌ FAIL${NC}: tmr.py has syntax errors"
    COMPILE_OK=false
fi

if python -m py_compile src/evaluation/eval_anonymization_v2.py 2>/dev/null; then
    echo -e "${GREEN}✅ PASS${NC}: eval_anonymization_v2.py compiles"
else
    echo -e "${RED}❌ FAIL${NC}: eval_anonymization_v2.py has syntax errors"
    COMPILE_OK=false
fi

if python -m py_compile src/evaluation/eval_model_main.py 2>/dev/null; then
    echo -e "${GREEN}✅ PASS${NC}: eval_model_main.py compiles"
else
    echo -e "${RED}❌ FAIL${NC}: eval_model_main.py has syntax errors"
    COMPILE_OK=false
fi

if python -m py_compile scripts/eval_same_action.py 2>/dev/null; then
    echo -e "${GREEN}✅ PASS${NC}: eval_same_action.py compiles"
else
    echo -e "${RED}❌ FAIL${NC}: eval_same_action.py has syntax errors"
    COMPILE_OK=false
fi
echo ""

# Test 6: Documentation exists
echo "┌──────────────────────────────────────────────────────────────────────────────┐"
echo "│ TEST 6: Documentation Files                                                  │"
echo "└──────────────────────────────────────────────────────────────────────────────┘"
echo ""

DOCS_OK=true

if [ -f "CLEANUP_COMPLETE.md" ]; then
    echo -e "${GREEN}✅ PASS${NC}: CLEANUP_COMPLETE.md exists"
else
    echo -e "${YELLOW}⚠️  WARN${NC}: CLEANUP_COMPLETE.md not found"
    DOCS_OK=false
fi

if [ -f "RETRAINING_PLAN.md" ]; then
    echo -e "${GREEN}✅ PASS${NC}: RETRAINING_PLAN.md exists"
else
    echo -e "${YELLOW}⚠️  WARN${NC}: RETRAINING_PLAN.md not found"
    DOCS_OK=false
fi

if [ -f "STATUS_REPORT.md" ]; then
    echo -e "${GREEN}✅ PASS${NC}: STATUS_REPORT.md exists"
else
    echo -e "${YELLOW}⚠️  WARN${NC}: STATUS_REPORT.md not found"
    DOCS_OK=false
fi
echo ""

# Summary
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                              ║"
echo "║                              SUMMARY                                         ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

if [ "$PY_FILES" -eq 1 ] && [ "$FILES_OK" = true ] && [ "$COMPILE_OK" = true ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
    echo ""
    echo "Repository cleanup is complete and verified."
    echo ""
    echo "Next steps:"
    echo "  1. Run: python tmr.py (select option 9 for status)"
    echo "  2. Run: python scripts/eval_same_action.py --dataset ntu_cv --num_pairs 100 --device cuda"
    echo "  3. View: open index.html"
    echo ""
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED${NC}"
    echo ""
    echo "Please review the failures above and fix them."
    echo ""
    exit 1
fi

