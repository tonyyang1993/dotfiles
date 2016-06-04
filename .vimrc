" Tony Yang's vim config

syntax enable
set tabstop=4           " 4 space tab
set expandtab           " use spaces for tabs
set softtabstop=4
set shiftwidth=4

filetype indent on
filetype plugin on
set autoindent

set number
set showcmd
set nocursorline
set relativenumber

set showmatch           " highlight matching parenthesis
set hlsearch
set incsearch

nnoremap H gT
nnoremap L gt
